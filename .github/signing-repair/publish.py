#!/usr/bin/env python3
"""Manual, content-bound GitHub-signed repair publisher; never writes a default branch."""
import base64, hashlib, json, os, re, sys, time, urllib.request
from pathlib import Path
API="https://api.github.com"
def api(method, path, body=None):
    data=None if body is None else json.dumps(body).encode()
    req=urllib.request.Request(API+path,data=data,method=method,headers={
        "Authorization":"Bearer "+os.environ["GITHUB_TOKEN"],
        "Accept":"application/vnd.github+json","X-GitHub-Api-Version":"2022-11-28",
        "Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=45) as r:
        raw=r.read()
        return json.loads(raw) if raw else None
def verify(sha, tree):
    for attempt in range(4):
        c=api("GET",f"/repos/{repo}/commits/{sha}")
        v=c["commit"]["verification"]
        if v.get("verified") is True and v.get("reason")=="valid":
            if c["commit"]["tree"]["sha"]!=tree:
                raise RuntimeError("Signed commit tree differs from reviewed tree")
            if [p["sha"] for p in c["parents"]]!=[p["base_sha"]]:
                raise RuntimeError("Signed commit ancestry differs from reviewed base")
            return c
        if attempt<3: time.sleep(2)
    raise RuntimeError("GitHub did not verify replacement commit: "+str(v.get("reason")))
raw=Path(".github/signing-repair/payload.json").read_bytes()
if hashlib.sha256(raw).hexdigest()!=os.environ["EXPECTED_PAYLOAD_SHA256"]:
    raise RuntimeError("Payload identity mismatch")
p=json.loads(raw)
repo=os.environ["GITHUB_REPOSITORY"]
if os.environ.get("GITHUB_EVENT_NAME")=="push" and os.environ.get("GITHUB_REF_NAME")!=os.environ["EXPECTED_BOOTSTRAP_BRANCH"]:
    raise RuntimeError("Unexpected bootstrap branch")
if p["repository"]!=repo or os.environ.get("GITHUB_EVENT_NAME") not in {"workflow_dispatch","push"}:
    raise RuntimeError("Unexpected repository or event")
if os.environ.get("GITHUB_ACTOR")!="dlworrell":
    raise RuntimeError("Owner dispatch required")
if not re.fullmatch(r"signing/[A-Za-z0-9._/-]+",p["branch"]) or ".." in p["branch"]:
    raise RuntimeError("Unsafe repair branch")
for k in ("base_sha","tree_sha"):
    if not re.fullmatch(r"[0-9a-f]{40}",p[k]): raise RuntimeError("Bad object identity")
metadata=api("GET",f"/repos/{repo}")
default=metadata["default_branch"]
if p["branch"]==default: raise RuntimeError("Default branch writes forbidden")
current=api("GET",f"/repos/{repo}/branches/{default}")
if current["commit"]["sha"]!=p["base_sha"]:
    raise RuntimeError("Default branch moved; reprepare without overwriting it")
seen=set()
for f in p["file_changes"].get("additions",[]):
    path=f["path"]
    if path.startswith("/") or ".." in path.split("/") or path in seen: raise RuntimeError("Unsafe or duplicate file path")
    seen.add(path)
    b=base64.b64decode(f["contents"],validate=True)
    actual=hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
    if actual!=f["blob_sha"]: raise RuntimeError("Payload file hash mismatch: "+path)
for f in p["file_changes"].get("deletions",[]):
    path=f["path"]
    if path.startswith("/") or ".." in path.split("/") or path in seen: raise RuntimeError("Unsafe deletion path")
    seen.add(path)
changes={kind:[{k:v for k,v in item.items() if k!="blob_sha"} for item in items] for kind,items in p["file_changes"].items()}
api("POST",f"/repos/{repo}/git/refs",{"ref":"refs/heads/"+p["branch"],"sha":p["base_sha"]})
mutation="mutation($input:CreateCommitOnBranchInput!){createCommitOnBranch(input:$input){commit{oid url}}}"
response=api("POST","/graphql",{"query":mutation,"variables":{"input":{
    "branch":{"repositoryNameWithOwner":repo,"branchName":p["branch"]},
    "expectedHeadOid":p["base_sha"],"message":p["message"],"fileChanges":changes}}})
if response.get("errors"): raise RuntimeError("GraphQL commit creation failed")
commit=response["data"]["createCommitOnBranch"]["commit"]
verified=verify(commit["oid"],p["tree_sha"])
result={"repository":repo,"branch":p["branch"],"commit":commit["oid"],"tree":p["tree_sha"],
        "base":p["base_sha"],"verified":True,"reason":"valid","url":commit["url"],
        "payload_sha256":hashlib.sha256(raw).hexdigest(),"files":len(seen),
        "signer":verified.get("committer",{}).get("login")}
Path("signing-result.json").write_text(json.dumps(result,indent=2)+"\n")
with open(os.environ["GITHUB_STEP_SUMMARY"],"a") as out:
    out.write("## Verified signing repair\n\n"+json.dumps(result,indent=2)+"\n")
print("SIGNING_REPAIR_RESULT="+json.dumps(result,sort_keys=True))
