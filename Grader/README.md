# Grader LMCache Scripts

This directory provides helper scripts to run disaggregated inference for the `gpt-120b-oss` model using [vLLM](https://docs.vllm.ai) with LMCache and the NIXL transfer layer. The workflow is split into two stages:

1. `prefill_disaggregated.py` performs the prefill phase on a GPU host and stores the KV cache inside LMCache using the high-throughput NIXL transport (ZeroMQ based).
2. `decode_disaggregated.py` attaches to the cached KV tensors from a different GPU host (or process) and performs decoding using the cached context.

Both scripts expose the same environment variable `GRADER_PROMPT` so that you can override the prompt without modifying the code. Run the scripts on two machines (or processes) that share the same ZeroMQ endpoints to simulate the full transfer. Each script prints detailed information about the ZeroMQ sockets, cache identifiers, and transfer lifecycle so that you can observe the flow of data in real time.

## Usage overview

1. **Prefill stage**

   ```bash
   python Grader/prefill_disaggregated.py \
     --control-endpoint tcp://0.0.0.0:5555 \
     --data-endpoint tcp://0.0.0.0:6000
   ```

   The prefill script will output a cache URI (for example `lmcache://grader/prefill/1234`) that must be provided to the decode stage.

2. **Decode stage**

   ```bash
   python Grader/decode_disaggregated.py \
     --control-endpoint tcp://prefill-host:5555 \
     --data-endpoint tcp://prefill-host:6000 \
     --cache-uri lmcache://grader/prefill/1234
   ```

   The decode script consumes the cache and completes the generation phase, showing how the KV cache is transferred over NIXL and reused by vLLM.

Both scripts enable vLLM prefix caching, chunked prefill, and standard optimizations by default. Adjust the arguments to experiment with different sampling parameters or LMCache compression strategies.
