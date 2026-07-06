import os, json

root = "./domainbed/output"

broken = []
good = []

for dirpath, dirnames, filenames in os.walk(root):
    if "results.json" in filenames:
        try:
            with open(os.path.join(dirpath, "results.json")) as f:
                json.load(f)
            good.append(dirpath)
        except Exception:
            broken.append(dirpath)

print("GOOD runs:", len(good))
print("BROKEN runs:", len(broken))

print("\nExample broken runs:")
for b in broken[:10]:
    print(b)