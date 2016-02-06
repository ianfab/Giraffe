### Giraffe ###

Giraffe is an experimental chess engine based on temporal-difference reinforcement learning with deep neural networks. It discovers almost all its chess knowledge through self-play.

For more information, see: http://arxiv.org/abs/1509.01549

Giraffe is written in C++11.

If you decide to compile Giraffe yourself, please grab the neural network definition files (eval.net and meval.net) from the binary distribution. They must be in Giraffe's working directory when Giraffe is started. Instructions on how to generate those files will be added later.

## Gaviota Tablebases ##
To use Gaviota tablebases, set the path through the GaviotaTbPath option.

To use Gaviota tablebases with the Wb2Uci adapter, set "GaviotaTbPath=..." in Wb2Uci.eng.

## Build ##
* The Makefile contains -ltcmalloc. libtcmalloc replaces malloc/free with another implementation with thread-local caching. It is optional and doesn't really matter for playing. It can be safely removed. Or you can install the library. It's in the libgoogle-perftools-dev package on Ubuntu (and probably other Debian-based distros).
* The Makefile contains -march=native. If you want to do a compile that also runs on older CPUs, change it to something else.
* Only GCC 4.8 or later is supported for now. Intel C/C++ Compiler can be easily supported by just changing compiler options. MSVC is not supported due to use of GCC intrinsics. Patches welcomed to provide alternate code path for MSVC. Clang is not supported due to lack of OpenMP.
* Tested on Linux (GCC 4.9), OS X (GCC 4.9), Windows (MinGW-W64 GCC 5.1). GCC versions earlier than 4.8 are definitely NOT supported, due to broken regex implementation in libstdc++.

## Training ##
Training Giraffe is a multi-step process that will take more than a week on a quad core machine if you want the highest quality results. Using a higher core count machine is recommended (about 3 days on a 20 cores Haswell Xeon).

1. Prepare a large file with many positions from real games. One line per position in FEN, that's all. No labels are needed. They don't have to be high quality games. I have extracted 5 million random positions from a CCRL dump. It's available for download on the Downloads page. Let's call this file ccrl.fen

2. Train the evaluation network:

```
#!bash
OMP_NUM_THREADS=n ./giraffe tdl ccrl.fen
```
where n is the number of cores you have.

It will periodically take a snapshot of the network, and store it in trainingResults/. You can stop the training (Ctrl-C) at any time.

Copy the latest file from trainingResults/ into the parent directory (where giraffe is), and rename it to eval.net.

It converges in about 72 hours on a 20-core Haswell Xeon (there is currently no automatic convergence detection, so you have to test snapshots periodically yourself, using whatever method you want - I used the STS).

3. Generate a database of inner nodes for training the move evaluator network:
This requires modifying the source code. Go to static_move_evaluator.h, and uncomment "//#define SAMPLING".

Then run:
```
#!bash
OMP_NUM_THREADS=n ./giraffe sample_internal ccrl.fen internal.fen
```

4. Label the internal positions for training the move evaluator network:
```
#!bash
OMP_NUM_THREADS=n ./giraffe label_bm internal.fen internal_labeled.fen
```

5. Finally, we can train the move evaluator network:
```
#!bash
OMP_NUM_THREADS=n ./giraffe train_move_eval internal_labeled.fen meval.net
```

That should give you meval.net in the end, and we are all done! eval.net is the position evaluation network, and meval.net is the move evaluation network. They should be in the working directory when giraffe is run. Run ./giraffe on the command line, and make sure that it says it's using the move evaluator network.