import json

packages = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "scikit-learn": "sklearn",
    "networkx": "networkx",
}

res = {}
for pkg_name, mod_name in packages.items():
    try:
        m = __import__(mod_name)
        ver = getattr(m, "__version__", None)
        res[pkg_name] = {"installed": True, "version": ver}
    except Exception as e:
        res[pkg_name] = {"installed": False, "error": str(e)}

print(json.dumps(res, ensure_ascii=False, indent=2))
