# Output of one run (CPU, fixed seeds)

## step1_fd.py
```
h(0,1):  European g = 0.3193 | hard stopping h_stop = 0.3336
  phi = 1      h_phi(0,1) = 0.3194   max lambda* = 0.1
  phi = 0.1    h_phi(0,1) = 0.3199   max lambda* = 0.8
  phi = 0.01   h_phi(0,1) = 0.3230   max lambda* = 4.3
  phi = 0.001  h_phi(0,1) = 0.3286   max lambda* = 14.1
ordering g <= h_1 <= h_0.1 <= h_0.01 <= h_0.001 <= h_stop holds everywhere: True
  phi = 1      max(h_stop - h_phi) = 0.0937   sqrt(2 phi r) = 0.4472
  phi = 0.1    max(h_stop - h_phi) = 0.0827   sqrt(2 phi r) = 0.1414
  phi = 0.01   max(h_stop - h_phi) = 0.0431   sqrt(2 phi r) = 0.0447
  phi = 0.001  max(h_stop - h_phi) = 0.0141   sqrt(2 phi r) = 0.0141
hard boundary x*(0) = 0.277
  refinement check phi=0.001: h(0,1) = 0.3286 (fine grid)
  refinement check phi=hard: h(0,1) = 0.3336 (fine grid)
```

## step2_sim.py
```
phi     h_FD(0,1)  MC value (s.e.)    stopped before T   mean tau | mean X_tau (stopped)   min (H-h) at stops
1       0.3194     0.3180 (0.0015)      0.4%             0.605    | 0.297                 9.9e-04
0.1     0.3199     0.3181 (0.0015)      4.0%             0.586    | 0.297                 1.2e-03
0.01    0.3230     0.3228 (0.0014)     22.1%             0.592    | 0.314                 2.3e-04
0.001   0.3286     0.3285 (0.0014)     44.1%             0.616    | 0.348                 6.8e-05
```

## step3_dgm.py 8000
```
phi      PDE loss  BC loss | h(0,1): DGM    FD   | max|DGM-FD| on x in [0.05,2.5] at t=0, 0.5, 0.9
1        1.4e-03   1.7e-07 |          0.3174 0.3194 |  0.0021  0.0019  0.0073   [107s]
0.1      7.1e-04   7.4e-08 |          0.3196 0.3199 |  0.0006  0.0013  0.0031   [213s]
0.01     4.2e-04   1.1e-07 |          0.3233 0.3230 |  0.0007  0.0010  0.0013   [320s]
0.001    3.6e-04   3.8e-07 |          0.3279 0.3286 |  0.0011  0.0007  0.0012   [424s]
```
