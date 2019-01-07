How to cuda-ize
===

## Selection

Gain: proportional to the selection scope

 - input: a grid of fitness
 - output: next_generation_reproducer
 - algorithm: some kind of 2d stencil

## Promoters around

 - input: a piece of DNA (size: 2 * promoter_size - 1)
 - output: an array of number of similarities (size: nb of offsets, at most promoter size)
 - algorithm: count similtudes, grid: offset, position

## Search terminator (`opt_prom_compute_RNA`)

 - input: starting pos, ?(all DNA ; just a bloc)
 - output: a variable firstFound
 - algorithm: loop with blocks (e.g size 64), kernel tries to find the first occurence of a terminator in the block, `atomicMin`

## Start protein (aka Shine Dal)

See terminator above -> but exact match
Also smallest blocks (e.g 32)
