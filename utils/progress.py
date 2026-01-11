# utils/progress.py

import sys
import time

class ProgressBar:
    def __init__(self, total, width=30, prefix="", leave=False):
        self.total = total
        self.width = width
        self.prefix = prefix
        self.leave =leave
        self.start_time = time.time()

    def update(self, step, postfix=""):
        """
        step: 当前完成的步数（从 1 开始）
        """
        ratio = step / self.total
        filled = int(self.width * ratio)
        bar = "=" * filled + ">" + "." * (self.width - filled - 1)

        elapsed = time.time() - self.start_time
        total_time = elapsed/(ratio)

        msg = (
            f"\r{self.prefix} "
            f"[{bar}] {step}/{self.total} "
            f"{ratio*100:5.1f}% "
            f"({elapsed:.1f}s /{total_time:.0f}s) {postfix}"
        )

        sys.stdout.write(msg)
        sys.stdout.flush()

    def close(self):
        if self.leave:
            sys.stdout.write("\r" + " " * (self.width + 80) + "\r")
        sys.stdout.write("\n")
