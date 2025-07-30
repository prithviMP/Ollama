#!/usr/bin/env python3
"""
Advanced Calculator Tool for Mathematical Operations

Provides comprehensive mathematical calculation capabilities for agents,
including basic arithmetic, scientific functions, and statistical operations.
"""

import math
import statistics
import re
from typing import Any, Optional, Dict, List
from pydantic import BaseModel, Field
from langchain.tools import BaseTool


class CalculatorTool(BaseTool):
    """Advanced calculator tool with scientific and statistical functions."""
    
    name = "calculator"
    description = """
    Perform mathematical calculations including:
    - Basic arithmetic: +, -, *, /, %, **
    - Scientific functions: sin, cos, tan, log, ln, sqrt, abs
    - Statistical operations: mean, median, mode, std, var
    - Constants: pi, e
    
    Input: Mathematical expression as string
    Examples: "2 + 3 * 4", "sqrt(16)", "mean([1,2,3,4,5])", "sin(pi/2)"
    Returns: Calculation result with explanation
    """
    
    def __init__(self):
        super().__init__()
        self.constants = {
            "pi": math.pi,
            "e": math.e,
            "inf": math.inf,
            "nan": math.nan
        }
        
        self.functions = {
            # Basic math
            "abs": abs,
            "round": round,
            "max": max,
            "min": min,
            "sum": sum,
            
            # Scientific functions
            "sqrt": math.sqrt,
            "pow": pow,
            "exp": math.exp,
            "log": math.log,
            "log10": math.log10,
            "ln": math.log,  # Natural log alias
            
            # Trigonometric
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
            "degrees": math.degrees,
            "radians": math.radians,
            
            # Statistical functions
            "mean": statistics.mean,
            "median": statistics.median,
            "mode": statistics.mode,
            "stdev": statistics.stdev,
            "variance": statistics.variance,
            "pstdev": statistics.pstdev,
            "pvariance": statistics.pvariance,
            
            # Advanced math
            "factorial": math.factorial,
            "gcd": math.gcd,
            "lcm": math.lcm,
            "floor": math.floor,
            "ceil": math.ceil,
        }
    
    def _run(self, expression: str) -> str:
        """Execute mathematical calculation."""
        try:
            # Clean and validate input
            expression = expression.strip()
            if not expression:
                return "Error: Empty expression provided"
            
            # Handle special function calls (like mean([1,2,3]))
            if self._is_function_call(expression):
                result = self._evaluate_function_call(expression)
            else:
                # Handle basic mathematical expressions
                result = self._evaluate_expression(expression)
            
            # Format result
            if isinstance(result, float):
                if result.is_integer():
                    result = int(result)
                else:
                    result = round(result, 10)  # Limit decimal places
            
            return f"Result: {result}\nExpression: {expression}"
            
        except ZeroDivisionError:
            return f"Error: Division by zero in expression: {expression}"
        except ValueError as e:
            return f"Error: Invalid value - {str(e)}"
        except TypeError as e:
            return f"Error: Type error - {str(e)}"
        except Exception as e:
            return f"Error: {str(e)} in expression: {expression}"
    
    def _is_function_call(self, expression: str) -> bool:
        """Check if expression is a function call."""
        # Look for patterns like function_name(args)
        function_pattern = r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        return bool(re.match(function_pattern, expression))
    
    def _evaluate_function_call(self, expression: str) -> Any:
        """Evaluate function calls like mean([1,2,3,4])."""
        # Extract function name and arguments
        match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)$', expression)
        if not match:
            raise ValueError(f"Invalid function call format: {expression}")
        
        func_name = match.group(1)
        args_str = match.group(2).strip()
        
        if func_name not in self.functions:
            raise ValueError(f"Unknown function: {func_name}")
        
        function = self.functions[func_name]
        
        # Parse arguments
        if not args_str:
            # No arguments
            return function()
        
        # Handle list arguments like [1,2,3,4]
        if args_str.startswith('[') and args_str.endswith(']'):
            # Parse list
            list_content = args_str[1:-1]
            if list_content.strip():
                args = [float(x.strip()) for x in list_content.split(',')]
                return function(args)
            else:
                return function([])
        
        # Handle multiple comma-separated arguments
        if ',' in args_str:
            args = [self._evaluate_expression(arg.strip()) for arg in args_str.split(',')]
            return function(*args)
        
        # Single argument
        arg = self._evaluate_expression(args_str)
        return function(arg)
    
    def _evaluate_expression(self, expression: str) -> float:
        """Evaluate basic mathematical expressions."""
        # Replace constants
        for const_name, const_value in self.constants.items():
            expression = expression.replace(const_name, str(const_value))
        
        # Create safe environment for eval
        safe_dict = {
            "__builtins__": {},
            "abs": abs,
            "round": round,
            "max": max,
            "min": min,
            "sum": sum,
            "pow": pow,
        }
        
        # Add math functions
        safe_dict.update(self.functions)
        safe_dict.update(self.constants)
        
        # Evaluate expression safely
        try:
            result = eval(expression, safe_dict, {})
            return result
        except:
            # Fallback to basic arithmetic evaluation
            return self._basic_arithmetic_eval(expression)
    
    def _basic_arithmetic_eval(self, expression: str) -> float:
        """Basic arithmetic evaluation as fallback."""
        # Remove spaces
        expression = expression.replace(' ', '')
        
        # Simple regex-based evaluation for basic operations
        # This is a simplified version - production code would use a proper parser
        
        # Handle parentheses first (simplified)
        while '(' in expression:
            # Find innermost parentheses
            start = expression.rfind('(')
            if start == -1:
                break
            end = expression.find(')', start)
            if end == -1:
                raise ValueError("Mismatched parentheses")
            
            inner = expression[start+1:end]
            inner_result = self._basic_arithmetic_eval(inner)
            expression = expression[:start] + str(inner_result) + expression[end+1:]
        
        # Handle exponentiation
        if '**' in expression:
            parts = expression.split('**')
            result = float(parts[0])
            for part in parts[1:]:
                result = result ** float(part)
            return result
        
        # Handle multiplication and division (left to right)
        if '*' in expression or '/' in expression:
            tokens = re.split(r'([*/])', expression)
            result = float(tokens[0])
            for i in range(1, len(tokens), 2):
                op = tokens[i]
                operand = float(tokens[i+1])
                if op == '*':
                    result *= operand
                else:  # op == '/'
                    result /= operand
            return result
        
        # Handle addition and subtraction (left to right)
        if '+' in expression or '-' in expression:
            # Handle negative numbers
            expression = expression.replace('+-', '-').replace('--', '+')
            
            tokens = re.split(r'([+-])', expression)
            result = float(tokens[0])
            for i in range(1, len(tokens), 2):
                op = tokens[i]
                operand = float(tokens[i+1])
                if op == '+':
                    result += operand
                else:  # op == '-'
                    result -= operand
            return result
        
        # Just a number
        return float(expression)
    
    async def _arun(self, expression: str) -> str:
        """Async version of the calculator."""
        return self._run(expression)


# Example usage and testing
def test_calculator():
    """Test the calculator tool."""
    calc = CalculatorTool()
    
    test_expressions = [
        "2 + 3 * 4",
        "sqrt(16)",
        "sin(pi/2)",
        "mean([1,2,3,4,5])",
        "max(10, 20, 5)",
        "factorial(5)",
        "log(100, 10)",
        "pow(2, 3)",
        "(2 + 3) * (4 - 1)",
        "abs(-42)",
        "round(3.14159, 2)"
    ]
    
    print("🧮 Calculator Tool Test Results:")
    print("=" * 50)
    
    for expr in test_expressions:
        result = calc._run(expr)
        print(f"Input: {expr}")
        print(f"Output: {result}")
        print("-" * 30)


if __name__ == "__main__":
    test_calculator()