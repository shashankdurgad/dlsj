| row | load s | median ms | p90 | p99 | speed-up vs baseline | actions/s (30-step chunk) | peak RSS MB | GPU MB | vs FP16 ref joints / gripper (deg) | demo err (ref) | power W |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fine-tuned 8k, OV GPU INT8+RTC (deployed) | 5.9 | 313.4 | 315.5 | 317.2 | n/a | 96 | 5898.0 | 1792.2 | 31.972 / 11.526 | 2.801 (32.626) | n/a |
| fine-tuned 5.5k, OV GPU INT8+RTC | 11.1 | 314.3 | 317.6 | 318.3 | n/a | 95 | 5858.1 | 1792.2 | 32.183 / 10.574 | 3.516 (32.626) | n/a |
| fine-tuned 8k wrist-edit, OV GPU INT8+RTC | 6.2 | 351.2 | 355.3 | 359.4 | n/a | 85 | 5899.4 | 1792.3 | 32.361 / 11.253 | 2.835 (32.626) | n/a |

RTC control-loop proxy: chunk 30 at 30 Hz, horizon 15: a new chunk is requested every 15 ticks (500 ms) and arrives after 313.4 ms (p99 317.2), i.e. 100% of requests land before the queue runs dry at p99; ticks are never idle while latency < 1000 ms.
