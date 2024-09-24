# Build the Package:
```
python setup.py sdist bdist_wheel
```

# Upload to PyPI Using Twine:
```
pip install twine
```

# Upload the package:
```
twine upload dist/*
```

**Note:**
- PyPI Account: You need to have a PyPI account. Create one here.
- Authentication: Configure ~/.pypirc with your PyPI credentials or use environment variables as per Twine's documentation.

**Uploading to a Private Repository:**
If you prefer a private repository (e.g., Nexus, Artifactory), adjust the Twine upload command accordingly by specifying the repository URL.

```twine upload --repository-url https://your-private-repo.com/simple/ dist/*```