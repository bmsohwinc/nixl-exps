## Tests
### get started test:
#### simple pytorch cuda p2p latency test
```sh
python tests/cuda_p2p_latency_baseline.py --src-device 0 --dst-device 1 --verify
```

#### simple nixl p2p latency test
```sh
# on terminal 1
python3 tests/basic_two_peers.py --mode=target --use_cuda=true --ip=127.0.0.1 --port=4242 

# on terminal 2
python3 tests/basic_two_peers.py --mode=initiator --use_cuda=true --ip=127.0.0.1 --port=4242
```
