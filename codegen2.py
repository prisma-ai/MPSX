import re

def generate_tensor_method(swift_code):
    # Extract function signature using regex
    signature_pattern = re.compile(r'func (\w+)\((.*: MPSGraphTensor,.*)(name: String\?\))')
    matches = signature_pattern.findall(swift_code)
    new_methods = ''

    for match in matches:
        # Remove the first and last parameters from the parameters list
        parameters = match[1].split(',')[1:-1]
        parameters = ', '.join(parameters).strip()

        # Generate the new method code
        new_methods += f'''
    func {match[0]}({parameters}) {{
        self.operation.graph.{match[0]}(self, {parameters}, name: nil)
    }}
        '''

    return new_methods

# Example of usage
swift_code = """
func exampleMethod(tensor: MPSGraphTensor, arg1: Int, arg2: Double, name: String?) {
    // Implementation...
}
"""

new_methods = generate_tensor_method(swift_code)
print(new_methods)