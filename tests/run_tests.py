import importlib
import traceback
import sys
import os

# Ensure project root is on sys.path so 'src' and 'tests' packages are importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TEST_MODULES = [
    'tests.test_train_helpers',
    'tests.test_chessai',
    'tests.test_app_endpoints'
]

def run():
    failures = 0
    for modname in TEST_MODULES:
        print(f'Running {modname}...')
        try:
            mod = importlib.import_module(modname)
        except Exception:
            print('  FAILED to import module')
            traceback.print_exc()
            failures += 1
            continue
        # call every function starting with 'test_'
        for name in dir(mod):
            if name.startswith('test_'):
                fn = getattr(mod, name)
                if callable(fn):
                    print(f'  {name}...', end=' ')
                    try:
                        fn()
                        print('OK')
                    except AssertionError as e:
                        print('FAIL')
                        traceback.print_exc()
                        failures += 1
                    except Exception:
                        print('ERROR')
                        traceback.print_exc()
                        failures += 1
    print('\nSummary: failures =', failures)
    if failures:
        raise SystemExit(1)

if __name__ == '__main__':
    run()
