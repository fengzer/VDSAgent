# Base System Message Template
DEBUG_TEMPLATE = """You are a professional code debugging expert, skilled at analyzing and fixing code errors.
"""

# Error Location Template
DEBUG_LOCATE_TEMPLATE = """Task: Locate Code Error
Task Description: Analyze code and error messages to locate the most relevant code snippet causing the error.

Input Data:
1. Error Code:
```python
{wrong_code}
```

2. Error Message:
{error_messages}

3. Output Message:
{output_messages}

Output Requirements:
1. Only output the code snippet causing the error
2. Do not attempt to fix the error at this stage
3. For assertion errors, find the underlying code causing the assertion failure
4. Output Format:
```python
[Relevant error code snippet]
```
"""

# Error Fix Template
DEBUG_FIX_TEMPLATE = """Task: Fix Code Error
Task Description: Fix the error code snippet based on error messages and code analysis.

Input Data:

1. Error Code Snippet:
```python
{most_relevant_code_snippet}
```

2. Error Message:
{error_messages}

3. Output Message:
{output_messages}


Output Requirements:
1. Analyze error cause
2. Provide fixed code
3. Ensure fixed code is compatible with predefined system tools
4. Output Format:
```python
[Fixed code snippet]
```
"""

# Code Merge Template
DEBUG_MERGE_TEMPLATE = """Task: Merge Fixed Code
Task Description: Integrate the fixed code snippet into the original code.

Input Data:
1. Original Code:
```python
{wrong_code}
```

2. Error Code Snippet:
```python
{most_relevant_code_snippet}
```

3. Fixed Code Snippet:
```python
{code_snippet_after_correction}
```

Output Requirements:
1. Replace the error part in original code with fixed code snippet
2. Maintain consistency in code structure and indentation
3. Output Format:
```python
[Complete fixed code]
```
"""

# Request Help Template
DEBUG_ASK_FOR_HELP_TEMPLATE = """Task: Evaluate Need for Human Assistance
Task Description: Analyze results of multiple debug attempts to determine if human assistance is needed.

Input Data:
1. Current Attempt Count: {i}
2. Historical Error Messages:
{all_error_messages}

Judgment Criteria:
1. Last two error messages are identical
2. More than two occurrences of same keywords in last three errors

Output Requirements:
1. If help needed, output: <HELP>Human assistance needed</HELP>
2. If help not needed, output empty string
"""

# Test Error Location Template
DEBUG_TEST_LOCATE_TEMPLATE = """Task: Locate Test Failure Code
Task Description: Analyze code and test failure information to locate the code snippet causing test failure.

Input Data:
1. Current Code:
```python
{code}
```

2. Test Failure Information:
{test_info}

3. Output Message:
{output_message}

Output Requirements:
1. Only output the code snippet causing test failure
2. Do not attempt to fix the issue at this stage
3. Output Format:
```python
[Relevant problem code snippet]
```
"""

# Test Error Fix Template
DEBUG_TEST_FIX_TEMPLATE = """Task: Fix Test Failure Code
Task Description: Fix the problem code snippet based on test failure information and code analysis.

Input Data:
1. Problem Code Snippet:
```python
{error_snippet}
```

2. Test Failure Information:
{test_info}

3. Output Message:
{output_message}

Output Requirements:
1. Analyze test failure cause
2. Provide fixed code
3. Ensure fixed code passes tests
4. Output Format:
```python
[Fixed code snippet]
```
"""

# Test Fix Code Merge Template
DEBUG_TEST_MERGE_TEMPLATE = """Task: Merge Test Fix Code
Task Description: Integrate the fixed code snippet into the original code.

Input Data:
1. Original Code:
```python
{original_code}
```

2. Problem Code Snippet:
```python
{error_snippet}
```

3. Fixed Code Snippet:
```python
{fixed_snippet}
```

Output Requirements:
1. Replace the problem part in original code with fixed code snippet
2. Maintain consistency in code structure and indentation
3. Ensure changes don't affect other functionality
4. Output Format:
```python
[Complete fixed code]
```
""" 