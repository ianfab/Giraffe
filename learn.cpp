/*
	Copyright (C) 2015 Matthew Lai

	Giraffe is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.
	
	Giraffe is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.
	
	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "learn.h"

#include <stdexcept>
#include <vector>
#include <sstream>
#include <algorithm>
#include <random>
#include <functional>

#include <cmath>

#include <omp.h>

#include "matrix_ops.h"
#include "board.h"
#include "ann/features_conv.h"
#include "ann/learn_ann.h"
#include "omp_scoped_thread_limiter.h"
#include "eval/eval.h"
#include "history.h"
#include "search.h"
#include "ttable.h"
#include "killer.h"
#include "countermove.h"
#include "random_device.h"
#include "ann/ann_evaluator.h"
#include "move_evaluator.h"
#include "static_move_evaluator.h"
#include "util.h"
#include "stats.h"

namespace
{

using namespace Learn;

std::string getFilename(int64_t iter)
{
	std::stringstream filenameSs;

	filenameSs << "trainingResults/eval" << iter << ".net";

	return filenameSs.str();
}

bool fileExists(const std::string &filename)
{
	std::ifstream is(filename);

	return is.good();
}

}

namespace Learn
{

void TDL(const std::string &positionsFilename)
{
	std::cout << "Starting TDL training..." << std::endl;

	std::ifstream positionsFile(positionsFilename);

	if (!positionsFile)
	{
		throw std::runtime_error(std::string("Cannot open ") + positionsFilename + " for reading");
	}

	// these are the root positions for training (they don't change)
	std::vector<std::string> rootPositions;

	std::string fen;

	std::cout << "Reading FENs..." << std::endl;

	while (std::getline(positionsFile, fen))
	{
		rootPositions.push_back(fen);
		assert(fen != "");
	}

	std::cout << "Positions read: " << rootPositions.size() << std::endl;

	int64_t numFeatures = FeaturesConv::GetNumFeatures();

	ANNEvaluator annEval;
	annEval.BuildANN(numFeatures);

	for (int64_t iteration = 0; iteration < NumIterations; ++iteration)
	{
		std::cout << "Iteration " << iteration << " ====================================" << std::endl;

		if (iteration == 0 && false)
		{
			auto rng = gRd.MakeMT();
			auto positionDist = std::uniform_int_distribution<size_t>(0, rootPositions.size() - 1);
			auto positionDrawFunc = std::bind(positionDist, rng);

			std::cout << "Bootstrapping using material eval" << std::endl;

			// first iteration is the bootstrap iteration where we don't do any TD, and simply use
			// material eval to bootstrap

			NNMatrixRM trainingBatch(PositionsFirstIteration, numFeatures);
			NNMatrixRM trainingTargets;

			trainingTargets.resize(trainingBatch.rows(), 1);

			std::vector<float> features;

			for (int64_t row = 0; row < trainingBatch.rows(); ++row)
			{
				Board b;
				Score val;

				do
				{
					b = rootPositions[positionDrawFunc()];
					val = Eval::gStaticEvaluator.EvaluateForWhite(b, SCORE_MIN, SCORE_MAX);
				} while (val == 0);

				FeaturesConv::ConvertBoardToNN(b, features);

				trainingBatch.block(row, 0, 1, trainingBatch.cols()) = MapStdVector(features);
				trainingTargets(row, 0) = Eval::gStaticEvaluator.UnScale(val);
			}

			for (size_t i = 0; i < 10; ++i)
			{
				EvalNet::Activations act;
				NNMatrixRM pred;

				for (int64_t start = 0; start < (trainingBatch.rows() - SGDBatchSize); start += SGDBatchSize)
				{
					auto xBlock = trainingBatch.block(start, 0, SGDBatchSize, trainingBatch.cols());
					auto targetsBlock = trainingTargets.block(start, 0, SGDBatchSize, 1);

					annEval.EvaluateForWhiteMatrix(xBlock, pred, act);

					float e = annEval.Train(pred, act, targetsBlock);

					UNUSED(e);

					#if 0
					if (start == 0)
					{
						std::cout << e << std::endl;
					}
					#endif
				}
			}
		}
		else
		{
			// a group of related positions (from the same root position)
			struct TrainingGroupInfo
			{
				std::vector<NNVector> leaves;

				enum class PositionType
				{
					EVAL, // the position's score matches eval of the leaf, and should be tuned
					FIXED // this is an EGTB or draw-by-rule position, and should not be tuned
				};

				std::vector<PositionType> positionTypes;

				std::vector<float> unscaledScores;

				int64_t GetSize() const { return static_cast<int64_t>(leaves.size()); }
			};

			std::vector<TrainingGroupInfo> trainingGroups;

			#pragma omp parallel
			{
				Killer killer;
				TTable ttable(1*MB); // we want the ttable to fit in L3
				CounterMove counter;
				History history;

				auto rng = gRd.MakeMT();
				auto positionDist = std::uniform_int_distribution<size_t>(0, rootPositions.size() - 1);
				auto positionDrawFunc = std::bind(positionDist, rng);

				// make a copy of the evaluator because evaluator is not thread-safe (due to caching)
				auto annEvalThread = annEval;

				std::vector<float> featureConvTemp;

				#pragma omp for schedule(dynamic, 1)
				for (int64_t batchPosNum = 0; batchPosNum < PositionsPerBatch; ++batchPosNum)
				{
					TrainingGroupInfo group;

					int64_t rootPosIdx = positionDrawFunc();
					Board pos = Board(rootPositions[rootPosIdx]);

					ttable.InvalidateAllEntries();

					if (pos.GetGameStatus() == Board::ONGOING)
					{
						// make 1 random move
						// it's very important that we make an odd number of moves, so that if the move is something stupid, the
						// opponent can take advantage of it (and we will learn that this position is bad) before we have a chance to
						// fix it
						MoveList ml;
						pos.GenerateAllLegalMoves<Board::ALL>(ml);

						auto movePickerDist = std::uniform_int_distribution<size_t>(0, ml.GetSize() - 1);

						pos.ApplyMove(ml[movePickerDist(rng)]);
					}

					// make a few moves, and store the leaves of each move into trainingBatch
					for (int64_t moveNum = 0; moveNum < HalfMovesToMake; ++moveNum)
					{
						if (pos.GetGameStatus() != Board::ONGOING)
						{
							break;
						}

						Search::SearchResult result = Search::SyncSearchNodeLimited(pos, SearchNodeBudget, &annEvalThread, &gStaticMoveEvaluator, &killer, &ttable, &counter, &history);

						Board leafPos = pos;
						leafPos.ApplyVariation(result.pv);

						Score rootScoreWhite = result.score * (pos.GetSideToMove() == WHITE ? 1 : -1);

						// this should theoretically be the same as the search result, except for mates, etc
						Score leafScore = annEvalThread.EvaluateForWhite(leafPos);

						TrainingGroupInfo::PositionType posType;

						if (result.pv.size() > 0 && (leafScore == rootScoreWhite))
						{
							posType = TrainingGroupInfo::PositionType::EVAL;
						}
						else
						{
							posType = TrainingGroupInfo::PositionType::FIXED;
						}

						group.unscaledScores.push_back(annEvalThread.UnScale(rootScoreWhite));
						group.positionTypes.push_back(posType);

						FeaturesConv::ConvertBoardToNN(leafPos, featureConvTemp);

						{
							NNVector featureVector = MapStdVector(featureConvTemp);
							group.leaves.push_back(std::move(featureVector));
						}

						if (posType == TrainingGroupInfo::PositionType::EVAL)
						{
							pos.ApplyMove(result.pv[0]);
							killer.MoveMade();
							ttable.AgeTable();
							history.NotifyMoveMade();
						}
						else
						{
							// if this is an end position already, don't make more moves
							break;
						}
					}

					#pragma omp critical(append_to_training_groups)
					{
						assert(group.leaves.size() == group.positionTypes.size());
						assert(group.leaves.size() == group.unscaledScores.size());

						trainingGroups.push_back(std::move(group));
					}
				}
			}

			int64_t totalNumPositions = 0;

			for (const auto &group : trainingGroups)
			{
				totalNumPositions += group.GetSize();
			}

			NNMatrixRM trainingBatch(totalNumPositions, numFeatures);
			NNMatrixRM pred(totalNumPositions, 1);
			NNMatrixRM targets(totalNumPositions, 1);

			// copy positions from position groups into one big matrix for performance
			int64_t currentRow = 0;
			for (const auto &group : trainingGroups)
			{
				for (const auto &leaf : group.leaves)
				{
					trainingBatch.block(currentRow, 0, 1, numFeatures) = leaf;
					++currentRow;
				}
			}

			assert(currentRow == totalNumPositions);

			EvalNet::Activations act;

			for (int64_t batchOptPass = 0; batchOptPass < OptimizationIterationsPerBatch; ++batchOptPass)
			{
				// for each pass, we -
				// 1. generate new predictions
				// 2. use TD to generate new targets
				// 3. do backprop using the new targets

				annEval.EvaluateForWhiteMatrix(trainingBatch, pred, act);

				int64_t currentRow = 0;

				for (const auto &group : trainingGroups)
				{
					for (int64_t currentPosition = 0; currentPosition < group.GetSize(); ++currentPosition)
					{
						float target = 0.0f;

						if (group.positionTypes[currentPosition] == TrainingGroupInfo::PositionType::FIXED)
						{
							// we have ground truth target for this position
							target = group.unscaledScores[currentPosition];
						}
						else
						{
							// do TD
							target = pred(currentRow, 0);

							float discount = TDLambda;

							float prevVal = target;

							for (int64_t futurePos = currentPosition + 1; futurePos < group.GetSize(); ++futurePos)
							{
								float val = 0.0f;
								// first we have to find out whether that position has a fixed score or not
								// if it does, we use that fixed score
								// otherwise, we use prediction

								if (group.positionTypes[futurePos] == TrainingGroupInfo::PositionType::FIXED)
								{
									val = group.unscaledScores[futurePos];
								}
								else
								{
									val = pred(currentRow + futurePos - currentPosition, 0);
								}

								float diff = val - prevVal;
								prevVal = val;

								target += diff * discount;
								discount *= TDLambda;
							}
						}

						targets(currentRow, 0) = target;
						++currentRow;
					}
				}

				float e = annEval.Train(pred, act, targets);
				UNUSED(e);
			}
		}

		if ((iteration % EvaluatorSerializeInterval) == 0)
		{
			std::cout << "Serializing " << getFilename(iteration) << "..." << std::endl;

			std::ofstream annOut(getFilename(iteration));

			annEval.Serialize(annOut);
		}
	}
}

}
