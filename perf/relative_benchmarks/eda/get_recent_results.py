# Copyright 2026 Syed Basim Ali
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import shutil

folders = ["../results/bench_op_f", "../results/bench_raw_buf"]
types = ["float64", "float32"]
cwd = os.getcwd()

for folder in folders:
    for t in types:
        pattern = os.path.join(folder, f"*_{t}_*_results.csv")
        files = glob.glob(pattern)
        if not files:
            print(f"[{t}] no files in {folder}")
            continue

        latest_file = max(files, key=os.path.getmtime)

        bn = os.path.basename(latest_file)
        idx = bn.find(t) + len(t)
        new_name = bn[:idx] + ".csv"
        dst = os.path.join(cwd, new_name)

        if not os.path.exists(dst) or os.path.getmtime(
            latest_file
        ) > os.path.getmtime(dst):
            shutil.copy2(latest_file, dst)
            print(f"Copied {latest_file} -> {dst}")
        else:
            print("Skipped copying, destination is up to date.")
