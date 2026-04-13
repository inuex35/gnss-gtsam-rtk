# gnss-fgo-rtk

RTK positioning using Factor Graph Optimization (GTSAM) with cssrlib observation model.

## Architecture

- **Observation model**: cssrlib (satellite positions, atmospheric corrections, cycle slip detection)
- **Optimizer**: GTSAM ISAM2 or IncrementalFixedLagSmoother
- **Factors**: C++ DDPseudorangeFactor / DDCarrierPhaseFactor (nonlinear)
- **Ambiguity resolution**: cssrlib LAMBDA + fix-and-hold

## Requirements

- GTSAM (built with DD factors)
- cssrlib >= 1.2
- numpy

## Usage

```bash
python examples/run_rtk_fls.py <rover.obs> <base.obs> <nav_file> <base_x> <base_y> <base_z>
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| LAG | 0 | IFLS lag (seconds), 0 = ISAM2 |
| SIG_PR | 0.3 | Pseudorange sigma (m) |
| SIG_CP | 0.003 | Carrier phase sigma (m) |
| SIG_DYN | 0.1 | Position dynamics sigma (m) |
| SIG_AMB | 30.0 | Initial ambiguity sigma (m) |
| HUBER_PR | 0 | Huber threshold for PR (0=off) |
| AR_MODE | 3 | 0:off, 1:continuous, 3:fix-and-hold |
| MAX_EP | all | Max epochs to process |
