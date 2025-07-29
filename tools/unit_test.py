from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
from langchain.base_language import BaseLanguageModel
import pandas as pd
import os
from typing import Dict, List, Tuple

class UnitTestInput(BaseModel):
    """Input parameter model"""
    phase: str = Field(..., description="Test phase (e.g., 'Data Cleaning', 'Feature Engineering', etc.)")
    data_path: str = Field(..., description="Path to cleaned data file")
    original_data_path: str = Field(None, description="Path to original data file (optional)")

class DataCleaningTest:
    """Collection of tests for data cleaning phase"""
    @staticmethod
    def test_file_readable(data_path: str) -> Dict:
        """Test if file is readable"""
        try:
            if not os.path.exists(data_path):
                return {
                    "name": "File Readability Test",
                    "passed": False,
                    "message": f"File does not exist: {data_path}"
                }
            
            pd.read_csv(data_path)
            return {
                "name": "File Readability Test",
                "passed": True,
                "message": "File exists and can be read correctly"
            }
        except Exception as e:
            return {
                "name": "File Readability Test",
                "passed": False,
                "message": f"Failed to read file: {str(e)}"
            }

    @staticmethod
    def test_empty_dataset(data_path: str) -> Dict:
        """Test if dataset is empty (only column names without content)"""
        try:
            df = pd.read_csv(data_path)
            if len(df) == 0:
                return {
                    "name": "Empty Dataset Test",
                    "passed": False,
                    "message": "Dataset has only column names, no actual content"
                }
            return {
                "name": "Empty Dataset Test",
                "passed": True,
                "message": f"Dataset contains {len(df)} rows"
            }
        except Exception as e:
            return {
                "name": "Empty Dataset Test",
                "passed": False,
                "message": f"Test execution failed: {str(e)}"
            }

    @staticmethod
    def test_missing_values(data_path: str) -> Dict:
        """Test for missing values"""
        try:
            df = pd.read_csv(data_path)
            missing_info = df.isnull().sum()
            missing_columns = missing_info[missing_info > 0]
            
            if missing_columns.empty:
                return {
                    "name": "Missing Values Test",
                    "passed": True,
                    "message": "No missing values in the data"
                }
            else:
                missing_details = []
                for col, count in missing_columns.items():
                    percentage = (count / len(df)) * 100
                    missing_details.append(f"{col}: {count} ({percentage:.2f}%)")
                
                return {
                    "name": "Missing Values Test",
                    "passed": False,
                    "message": f"Found missing values:\n" + "\n".join(missing_details)
                }
        except Exception as e:
            return {
                "name": "Missing Values Test",
                "passed": False,
                "message": f"Test execution failed: {str(e)}"
            }

    @staticmethod
    def test_duplicated_features(data_path: str) -> Dict:
        """Test for duplicate feature columns"""
        try:
            df = pd.read_csv(data_path)
            duplicated_features = df.columns[df.columns.duplicated()].tolist()
            
            if not duplicated_features:
                return {
                    "name": "Duplicate Features Test",
                    "passed": True,
                    "message": "No duplicate feature columns"
                }
            else:
                return {
                    "name": "Duplicate Features Test",
                    "passed": False,
                    "message": f"Found duplicate feature columns: {', '.join(duplicated_features)}"
                }
        except Exception as e:
            return {
                "name": "Duplicate Features Test",
                "passed": False,
                "message": f"Test execution failed: {str(e)}"
            }

    @staticmethod
    def test_data_consistency(data_path: str, original_path: str) -> Dict:
        """Test data consistency between cleaned and original data"""
        try:
            cleaned_df = pd.read_csv(data_path)
            original_df = pd.read_csv(original_path)
            
            # Check row count
            if len(cleaned_df) > len(original_df):
                return {
                    "name": "Data Consistency Test",
                    "passed": False,
                    "message": f"Cleaned data row count ({len(cleaned_df)}) exceeds original data row count ({len(original_df)})"
                }
            
            # Check common columns
            common_columns = set(original_df.columns) & set(cleaned_df.columns)
            if not common_columns:
                return {
                    "name": "Data Consistency Test",
                    "passed": False,
                    "message": "No common columns between cleaned and original data"
                }
            
            return {
                "name": "Data Consistency Test",
                "passed": True,
                "message": f"Data consistency check passed, retained {len(common_columns)} original features"
            }
        except Exception as e:
            return {
                "name": "Data Consistency Test",
                "passed": False,
                "message": f"Test execution failed: {str(e)}"
            }

    @staticmethod
    def test_duplicated_rows(data_path: str) -> Dict:
        """Test for duplicate rows"""
        try:
            df = pd.read_csv(data_path)
            duplicated_count = df.duplicated().sum()
            
            if duplicated_count == 0:
                return {
                    "name": "Duplicate Rows Test",
                    "passed": True,
                    "message": "No duplicate rows"
                }
            else:
                # Get indices of duplicate rows
                duplicated_rows = df[df.duplicated(keep='first')].index.tolist()
                return {
                    "name": "Duplicate Rows Test",
                    "passed": False,
                    "message": f"Found {duplicated_count} duplicate rows, indices: {duplicated_rows}"
                }
        except Exception as e:
            return {
                "name": "Duplicate Rows Test",
                "passed": False,
                "message": f"Test execution failed: {str(e)}"
            }

    @staticmethod
    def test_numeric_conversion(data_path: str) -> Dict:
        """Test if all categorical variables have been converted to numeric"""
        try:
            df = pd.read_csv(data_path)
            non_numeric_cols = df.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
            
            if not non_numeric_cols:
                return {
                    "name": "Numeric Conversion Test",
                    "passed": True,
                    "message": "All variables are numeric"
                }
            else:
                return {
                    "name": "Numeric Conversion Test",
                    "passed": False,
                    "message": f"Found non-numeric variables: {', '.join(non_numeric_cols)}"
                }
        except Exception as e:
            return {
                "name": "Numeric Conversion Test",
                "passed": False,
                "message": f"Test execution failed: {str(e)}"
            }

    @staticmethod
    def test_data_retention(data_path: str, original_path: str) -> Dict:
        """Test if cleaned dataset retains sufficient samples (at least 85%)"""
        try:
            cleaned_df = pd.read_csv(data_path)
            original_df = pd.read_csv(original_path)
            
            # Calculate data retention ratio
            retention_ratio = len(cleaned_df) / len(original_df) * 100
            
            if retention_ratio >= 85:
                return {
                    "name": "Data Retention Test",
                    "passed": True,
                    "message": f"Data retention rate is {retention_ratio:.2f}%, meets requirement (≥85%)"
                }
            else:
                return {
                    "name": "Data Retention Test",
                    "passed": False,
                    "message": f"Data retention rate is {retention_ratio:.2f}%, below required 85%. Original data: {len(original_df)} rows, Cleaned data: {len(cleaned_df)} rows"
                }
        except Exception as e:
            return {
                "name": "Data Retention Test",
                "passed": False,
                "message": f"Test execution failed: {str(e)}"
            }

class UnitTestTool(BaseTool):
    name = "unit_test"
    description = "Execute unit tests for different phases"
    args_schema: type[BaseModel] = UnitTestInput
    
    # Define test function mapping for each phase
    PHASE_TESTS = {
        "Data Cleaning": [
            DataCleaningTest.test_file_readable,
            DataCleaningTest.test_empty_dataset,
            DataCleaningTest.test_missing_values,
            DataCleaningTest.test_duplicated_features,
            DataCleaningTest.test_duplicated_rows,
            DataCleaningTest.test_data_consistency,
            DataCleaningTest.test_data_retention,
            #DataCleaningTest.test_numeric_conversion,
        ]
    }
    
    def _generate_report(self, results: Dict) -> str:
        """Generate test report"""
        report = [f"Test Report for {results['phase']}:"]
        report.append(f"Passed: {results['passed']}, Failed: {results['failed']}\n")
        
        for test in results["tests"]:
            status = "✓" if test["passed"] else "✗"
            report.append(f"{status} {test['name']}: {test['message']}")
            
        return "\n".join(report)

    def _run(self, phase: str, data_path: str, original_data_path: str = None) -> Dict:
        """Run unit tests for specified phase"""
        results = {
            "phase": phase,
            "tests": [],
            "passed": 0,
            "failed": 0,
            "all_passed": False,
            "report": ""
        }
        
        # Get test functions list for current phase
        test_functions = self.PHASE_TESTS.get(phase, [])
        if not test_functions:
            results["tests"].append({
                "name": "Phase Check",
                "passed": False,
                "message": f"Unimplemented test phase: {phase}"
            })
            results["failed"] = 1
            results["report"] = self._generate_report(results)
            return results
            
        # Execute tests
        for test_func in test_functions:
            # List of test functions that need original data path
            needs_original_data = [
                DataCleaningTest.test_data_consistency,
                DataCleaningTest.test_data_retention
            ]
            
            if test_func in needs_original_data and original_data_path:
                test_result = test_func(data_path, original_data_path)
            else:
                test_result = test_func(data_path)
                
            results["tests"].append(test_result)
            if test_result["passed"]:
                results["passed"] += 1
            else:
                results["failed"] += 1
        
        # Check if all tests passed
        results["all_passed"] = results["failed"] == 0
        # Generate report
        results["report"] = self._generate_report(results)
                
        return results