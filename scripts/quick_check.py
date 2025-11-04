#!/usr/bin/env python3
"""Quick system validation script for Krator trading system.

This script performs comprehensive checks of all system dependencies,
configurations, and services to ensure the system is ready to run.
"""

import sys
import os
import importlib
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from config.settings import get_settings
except ImportError:
    print("❌ Error: Cannot import settings. Run from project root directory.")
    sys.exit(1)


class SystemValidator:
    """System validation and health checker."""
    
    def __init__(self):
        """Initialize system validator."""
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = 0
        self.results: List[Tuple[str, bool, str]] = []
        
        # Try to load settings
        try:
            self.settings = get_settings()
        except Exception as e:
            self.settings = None
            self.results.append(("Settings Loading", False, f"Failed to load settings: {e}"))
    
    def check_result(self, name: str, success: bool, message: str, warning: bool = False) -> None:
        """Record check result.
        
        Args:
            name: Check name
            success: Whether check passed
            message: Result message
            warning: Whether this is a warning rather than failure
        """
        if success:
            self.checks_passed += 1
            print(f"✅ {name}: {message}")
        elif warning:
            self.warnings += 1
            print(f"⚠️ {name}: {message}")
        else:
            self.checks_failed += 1
            print(f"❌ {name}: {message}")
        
        self.results.append((name, success, message))
    
    def check_python_version(self) -> None:
        """Check Python version compatibility."""
        version = sys.version_info
        
        if version.major == 3 and version.minor >= 11:
            self.check_result(
                "Python Version", 
                True, 
                f"Python {version.major}.{version.minor}.{version.micro} is supported"
            )
        elif version.major == 3 and version.minor >= 8:
            self.check_result(
                "Python Version", 
                True, 
                f"Python {version.major}.{version.minor}.{version.micro} works but 3.11+ recommended",
                warning=True
            )
        else:
            self.check_result(
                "Python Version", 
                False, 
                f"Python {version.major}.{version.minor}.{version.micro} is not supported. Need 3.8+"
            )
    
    def check_talib_installation(self) -> None:
        """Check TA-Lib installation and functionality."""
        try:
            import talib
            import numpy as np
            
            # Test basic functionality
            test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 2.0] * 5, dtype=float)
            sma_result = talib.SMA(test_data, timeperiod=5)
            
            if len(sma_result) == len(test_data) and not all(np.isnan(sma_result[-5:])):
                self.check_result(
                    "TA-Lib Installation", 
                    True, 
                    f"TA-Lib {talib.__version__} working correctly"
                )
            else:
                self.check_result(
                    "TA-Lib Installation", 
                    False, 
                    "TA-Lib installed but not functioning correctly"
                )
                
        except ImportError:
            self.check_result(
                "TA-Lib Installation", 
                False, 
                "TA-Lib not installed. Run: brew install ta-lib && pip install ta-lib"
            )
        except Exception as e:
            self.check_result(
                "TA-Lib Installation", 
                False, 
                f"TA-Lib error: {e}"
            )
    
    def check_core_dependencies(self) -> None:
        """Check core Python dependencies."""
        required_packages = [
            ('pandas', 'pandas'),
            ('numpy', 'numpy'),
            ('loguru', 'loguru'),
            ('pydantic', 'pydantic'),
            ('sqlalchemy', 'SQLAlchemy'),
            ('redis', 'redis'),
            ('celery', 'celery'),
            ('psutil', 'psutil'),
            ('aiohttp', 'aiohttp'),
        ]
        
        missing_packages = []
        working_packages = []
        
        for package_name, import_name in required_packages:
            try:
                module = importlib.import_module(import_name)
                version = getattr(module, '__version__', 'unknown')
                working_packages.append(f"{package_name} ({version})")
            except ImportError:
                missing_packages.append(package_name)
        
        if not missing_packages:
            self.check_result(
                "Core Dependencies", 
                True, 
                f"All {len(working_packages)} core packages installed"
            )
        else:
            self.check_result(
                "Core Dependencies", 
                False, 
                f"Missing packages: {', '.join(missing_packages)}"
            )
    
    def check_database_connection(self) -> None:
        """Check database connectivity."""
        if not self.settings:
            self.check_result("Database Connection", False, "Settings not loaded")
            return
        
        try:
            from sqlalchemy import create_engine, text
            
            engine = create_engine(
                self.settings.database.url,
                pool_pre_ping=True,
                connect_args={"connect_timeout": 5}
            )
            
            with engine.connect() as conn:
                result = conn.execute(text("SELECT version()"))
                version_info = result.fetchone()[0]
                
            self.check_result(
                "Database Connection", 
                True, 
                f"Connected to PostgreSQL: {version_info[:50]}..."
            )
            
        except Exception as e:
            self.check_result(
                "Database Connection", 
                False, 
                f"Database connection failed: {e}"
            )
    
    def check_redis_connection(self) -> None:
        """Check Redis connectivity."""
        if not self.settings:
            self.check_result("Redis Connection", False, "Settings not loaded")
            return
        
        try:
            import redis
            
            r = redis.from_url(
                self.settings.redis.url,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test basic operations
            r.ping()
            info = r.info()
            redis_version = info['redis_version']
            
            self.check_result(
                "Redis Connection", 
                True, 
                f"Connected to Redis {redis_version}"
            )
            
        except Exception as e:
            self.check_result(
                "Redis Connection", 
                False, 
                f"Redis connection failed: {e}"
            )
    
    def check_environment_variables(self) -> None:
        """Check critical environment variables."""
        if not self.settings:
            self.check_result("Environment Variables", False, "Settings not loaded")
            return
        
        critical_vars = [
            ('DATABASE_URL', self.settings.database.url),
            ('REDIS_URL', self.settings.redis.url),
        ]
        
        optional_vars = [
            ('ALPACA_API_KEY', getattr(self.settings.alpaca, 'api_key', None)),
            ('ALPACA_SECRET_KEY', getattr(self.settings.alpaca, 'secret_key', None)),
            ('SLACK_WEBHOOK_URL', getattr(self.settings.slack, 'webhook_url', None)),
        ]
        
        # Check critical variables
        missing_critical = []
        for var_name, var_value in critical_vars:
            if not var_value or var_value.startswith('change_this') or 'localhost' in var_value:
                missing_critical.append(var_name)
        
        if not missing_critical:
            self.check_result(
                "Critical Environment Variables", 
                True, 
                "All critical environment variables configured"
            )
        else:
            self.check_result(
                "Critical Environment Variables", 
                False, 
                f"Missing or default values: {', '.join(missing_critical)}"
            )
        
        # Check optional variables
        missing_optional = []
        for var_name, var_value in optional_vars:
            if not var_value or var_value.startswith('your_') or var_value.startswith('change_'):
                missing_optional.append(var_name)
        
        if missing_optional:
            self.check_result(
                "Optional Environment Variables", 
                True, 
                f"Consider configuring: {', '.join(missing_optional)}",
                warning=True
            )
        else:
            self.check_result(
                "Optional Environment Variables", 
                True, 
                "All optional variables configured"
            )
    
    def check_project_structure(self) -> None:
        """Check project directory structure."""
        required_dirs = [
            'core',
            'data', 
            'infra',
            'config',
            'db',
            'tests'
        ]
        
        required_files = [
            'core/events.py',
            'core/engine.py',
            'data/indicators.py',
            'infra/logging_config.py',
            'infra/alerts.py',
            'config/settings.py',
            'db/models.py',
            'requirements.txt',
            '.env.example'
        ]
        
        missing_dirs = []
        missing_files = []
        
        # Check directories
        for dir_name in required_dirs:
            if not (project_root / dir_name).is_dir():
                missing_dirs.append(dir_name)
        
        # Check files
        for file_name in required_files:
            if not (project_root / file_name).is_file():
                missing_files.append(file_name)
        
        if not missing_dirs and not missing_files:
            self.check_result(
                "Project Structure", 
                True, 
                f"All {len(required_dirs)} directories and {len(required_files)} files present"
            )
        else:
            issues = []
            if missing_dirs:
                issues.append(f"Missing dirs: {', '.join(missing_dirs)}")
            if missing_files:
                issues.append(f"Missing files: {', '.join(missing_files)}")
            
            self.check_result(
                "Project Structure", 
                False, 
                "; ".join(issues)
            )
    
    def check_docker_availability(self) -> None:
        """Check Docker installation and availability."""
        try:
            result = subprocess.run(
                ['docker', '--version'], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            
            if result.returncode == 0:
                version = result.stdout.strip()
                self.check_result(
                    "Docker Availability", 
                    True, 
                    f"{version}"
                )
            else:
                self.check_result(
                    "Docker Availability", 
                    False, 
                    "Docker command failed"
                )
                
        except subprocess.TimeoutExpired:
            self.check_result(
                "Docker Availability", 
                False, 
                "Docker command timed out"
            )
        except FileNotFoundError:
            self.check_result(
                "Docker Availability", 
                True, 
                "Docker not installed (optional for development)",
                warning=True
            )
        except Exception as e:
            self.check_result(
                "Docker Availability", 
                False, 
                f"Docker check failed: {e}"
            )
    
    def check_import_system(self) -> None:
        """Check that core system modules can be imported."""
        core_modules = [
            'config.settings',
            'core.events', 
            'core.engine',
            'data.indicators',
            'infra.logging_config',
            'infra.alerts',
            'db.models'
        ]
        
        import_errors = []
        imported_modules = []
        
        for module_name in core_modules:
            try:
                importlib.import_module(module_name)
                imported_modules.append(module_name)
            except ImportError as e:
                import_errors.append(f"{module_name}: {e}")
            except Exception as e:
                import_errors.append(f"{module_name}: {type(e).__name__}: {e}")
        
        if not import_errors:
            self.check_result(
                "Module Imports", 
                True, 
                f"All {len(imported_modules)} core modules import successfully"
            )
        else:
            self.check_result(
                "Module Imports", 
                False, 
                f"Import errors: {'; '.join(import_errors)}"
            )
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all system validation checks.
        
        Returns:
            Dictionary with check results and summary
        """
        print("\n🔍 Krator System Validation\n")
        print("=" * 50)
        
        # Run all checks
        self.check_python_version()
        self.check_talib_installation()
        self.check_core_dependencies()
        self.check_project_structure()
        self.check_import_system()
        self.check_environment_variables()
        self.check_database_connection()
        self.check_redis_connection()
        self.check_docker_availability()
        
        # Print summary
        print("\n" + "=" * 50)
        print(f"\n📊 Summary:")
        print(f"✅ Passed: {self.checks_passed}")
        print(f"❌ Failed: {self.checks_failed}")
        print(f"⚠️ Warnings: {self.warnings}")
        
        if self.checks_failed == 0:
            print("\n🎉 All critical checks passed! System is ready to run.")
            if self.warnings > 0:
                print(f"⚠️ Note: {self.warnings} warnings should be addressed for optimal operation.")
        else:
            print(f"\n❌ {self.checks_failed} critical issues must be resolved before running the system.")
        
        print("\n" + "=" * 50 + "\n")
        
        return {
            'checks_passed': self.checks_passed,
            'checks_failed': self.checks_failed,
            'warnings': self.warnings,
            'results': self.results,
            'ready': self.checks_failed == 0
        }


def main():
    """Main entry point for quick check script."""
    validator = SystemValidator()
    results = validator.run_all_checks()
    
    # Exit with appropriate code
    if results['ready']:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure


if __name__ == "__main__":
    main()
