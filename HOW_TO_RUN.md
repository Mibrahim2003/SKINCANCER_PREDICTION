# How to Run the Skin Cancer Model Training Pipeline

## ⚠️ **IMPORTANT: Why the Import Error Happens**

When you run `python workflow.py` from inside the `app/` directory, you get:
```
ModuleNotFoundError: No module named 'app'
```

**Why?** The workflow.py file has imports like:
```python
from app.utils import load_artifacts
from app.ml_validation import validate_training_run
```

Python looks for the `app` module starting from:
1. The directory you're running from
2. Directories in `sys.path` (Python path)

When you run from **inside** the `app/` directory, Python can't find `app` because `app` IS the current directory, not a parent module.

When you run the `.bat` file, it works because:
1. It changes to the project root (`c:\Users\ibrah\Desktop\New Project`)
2. It sets `PYTHONPATH` to the project root
3. From the project root, Python can find the `app` module

---

## ✅ **CORRECT WAYS TO RUN THE PIPELINE**

### **Method 1: Using run.bat (RECOMMENDED - Always Works)**
```bash
# From ANY directory, just double-click:
run.bat

# Or from terminal:
run.bat 100        # Train with 100 samples
run.bat 500        # Train with 500 samples
```

**Advantages:**
- ✅ Works from any directory
- ✅ Automatically activates virtual environment
- ✅ Sets correct Python path
- ✅ No configuration needed

---

### **Method 2: Using run_pipeline.py (New Universal Runner)**
```bash
# From project root only:
cd "c:\Users\ibrah\Desktop\New Project"
.venv\Scripts\activate
python run_pipeline.py 100
```

**Advantages:**
- ✅ Python-based (cross-platform)
- ✅ Automatically sets correct paths
- ✅ Clean Python syntax

**Disadvantage:**
- ❌ Must activate venv manually
- ❌ Must be at project root

---

### **Method 3: Direct Execution (Now Fixed!)**
```bash
# From project root:
cd "c:\Users\ibrah\Desktop\New Project"
.venv\Scripts\activate
python app\workflow.py 100
```

**Why it works now:**
The workflow.py file now has this code at the top:
```python
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
```

This dynamically adds the project root to Python's path.

**Advantages:**
- ✅ Direct control
- ✅ Works from project root
- ✅ Standard Python execution

**Disadvantage:**
- ❌ Must activate venv manually
- ❌ Must be at project root

---

### **❌ WRONG WAYS (Will Fail)**

```bash
# DON'T DO THIS - Will fail with ModuleNotFoundError:
cd app
python workflow.py 100
```

**Why it fails:** Even with the fix, running from inside `app/` directory causes issues because Python can't resolve relative imports properly.

---

## 🎯 **QUICK REFERENCE**

| Method | Command | Need venv? | Works from anywhere? | Recommended? |
|--------|---------|------------|---------------------|--------------|
| **run.bat** | `run.bat 100` | No (auto) | ✅ Yes | ⭐ **BEST** |
| **run_pipeline.py** | `python run_pipeline.py 100` | Yes | ❌ Root only | Good |
| **Direct** | `python app\workflow.py 100` | Yes | ❌ Root only | Advanced |

---

## 📊 **After Running**

1. **Check the output** in your terminal for task progress
2. **Open validation report**: `reports\validation_report.html`
3. **Review trained model**: Saved in `models\` directory
4. **Check Discord** for notification (if configured)

---

## 🔧 **Troubleshooting**

### "ModuleNotFoundError: No module named 'prefect'"
**Solution:** Activate virtual environment first:
```bash
.venv\Scripts\activate
```

### "ModuleNotFoundError: No module named 'app'"
**Solution:** Use `run.bat` OR make sure you're at project root:
```bash
cd "c:\Users\ibrah\Desktop\New Project"
```

### "Cannot find the path specified"
**Solution:** Check that you're in the project directory:
```bash
cd "c:\Users\ibrah\Desktop\New Project"
dir  # Should see: app/, models/, requirements.txt, etc.
```

---

## 🎓 **Understanding Python Imports**

The project structure:
```
New Project/          ← Project root
├── app/
│   ├── workflow.py   ← Has "from app.utils import ..."
│   ├── utils.py
│   └── ml_validation.py
├── run.bat
└── run_pipeline.py
```

When Python sees `from app.utils import ...`, it looks for:
- A folder named `app` 
- Starting from directories in `sys.path`

**From project root:** ✅ `app/` is visible → imports work
**From app/ directory:** ❌ `app/` is the current dir, not a module → imports fail

This is why **run.bat** is the safest option - it always runs from the correct location.

---

## 🚀 **My Recommendation**

**Just use `run.bat`** - it handles everything automatically:
- ✅ Activates virtual environment
- ✅ Sets correct paths
- ✅ Runs from correct directory
- ✅ Works every time

Double-click it or run from terminal: `run.bat 100`

Done! 🎉
