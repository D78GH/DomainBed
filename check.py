import json
f=open('domainbed/output/1cfbdea844370663f7c14836674c0ae5/results.jsonl')
print(json.loads(f.readlines()[-1]).keys())