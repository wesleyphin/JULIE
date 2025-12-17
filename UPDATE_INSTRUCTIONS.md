# How to Update Your Local Julie-UI

You're seeing a `ModuleNotFoundError` because you need to pull the latest changes.

## On Your Mac:

```bash
cd /Users/whosvxn/Downloads/JULIE

# Make sure you're on the correct branch
git checkout claude/add-account-selection-ui-PQjq0

# Pull the latest changes
git pull origin claude/add-account-selection-ui-PQjq0

# Verify the files are there
ls -la account_selector.py julie_ui.py

# Now run Julie-UI
python3 julie_ui.py
```

## Required Files

Make sure these files are present after pulling:
- ✅ `account_selector.py` (NEW - the account selection UI)
- ✅ `julie_ui.py` (RENAMED from monitor_ui.py)
- ✅ `terminal_ui.py` (UPDATED)
- ✅ `client.py` (UPDATED)
- ✅ `ACCOUNT_SELECTION_GUIDE.md` (NEW - documentation)

## If Git Pull Doesn't Work

If you get merge conflicts or issues, you can:

1. **Stash any local changes:**
   ```bash
   git stash
   ```

2. **Pull again:**
   ```bash
   git pull origin claude/add-account-selection-ui-PQjq0
   ```

3. **Or download the files directly from GitHub:**
   - Go to: https://github.com/wesleyphin/JULIE/tree/claude/add-account-selection-ui-PQjq0
   - Download the updated files

## Dependencies

Also make sure you have the Rich library installed:
```bash
pip3 install rich
```

## Quick Test

After updating, verify it works:
```bash
python3 julie_ui.py
```

You should see the beautiful account selection interface!
