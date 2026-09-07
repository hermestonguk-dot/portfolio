# Uploading The Portfolio To GitHub

The portfolio is already a local Git repository on the `main` branch. The
repository is connected to `origin`, so the normal upload is a commit followed
by a push.

## 1. Review the files

From PowerShell:

```powershell
cd "C:\Users\Hermes\UOM CV\portfolio"
git status
git diff --stat
```

Open the main page locally by opening `index.html` in a browser. Check the
project links and read the README files for the projects you plan to show.

## 2. Check for private material

Before making the repository public, search for personal notes and credentials:

```powershell
rg -n -i "password|secret|api[_-]?key|token" .
```

Review the university coursework publication notes. Keep the repository
private if a module does not permit public sharing of submitted solutions.

## 3. Commit the cleaned portfolio

```powershell
git add .
git status
git commit -m "Curate MSc project portfolio"
```

The `git status` step lets you check the staged file list before the commit.

## 4. Sign in and push

Check the remote:

```powershell
git remote -v
```

Then push the `main` branch:

```powershell
git push -u origin main
```

Git Credential Manager may open a browser for GitHub sign-in. Complete that
sign-in with the GitHub account that owns the repository.

## 5. Enable GitHub Pages

On GitHub, open the repository and choose **Settings**, **Pages**, and **Deploy
from a branch**. Select the `main` branch and the `/ (root)` folder, then save.
The site will be available at the Pages URL shown by GitHub after deployment.

## 6. Later updates

After changing files:

```powershell
git add .
git commit -m "Update portfolio evidence"
git push
```

The project pages are ordinary Markdown and source files, so recruiters can
read the evidence even if the GitHub Pages site is not enabled.
