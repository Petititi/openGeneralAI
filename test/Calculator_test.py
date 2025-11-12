
class Calculator:
    """A simple calculator"""
    
    def add(self, a, b):
        """Add two numbers"""
        return a + b
    
    def subtract(self, a, b):
        """Subtract two numbers"""
        return a - b
    
    def multiply(self, a, b):
        """Multiply two numbers"""
        return a * b
    
    def divide(self, a, b):
        """Divide two numbers"""
        if b == 0:
            raise ValueError("Division by zero")
        return a / b

def standalone_function():
    """A standalone function"""
    print("Hello World")