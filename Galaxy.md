# Install Planemo
```
python -m venv planemo
. planemo/bin/activate
pip install planemo
```

### Initialize chap.xml
```
planemo tool_init --id 'chap' --name 'CHESS Analysis Pipeline' --example_command 'runner.py --config config.yaml' --example_input config.yaml --cite_url 'https://github.com/CHESSComputing/ChessAnalysisPipeline' --help_from_command 'runner.py --help' --test_case --example_output data.out
```
this command will output:
```
Tool written to chap.xml
No test-data directory, creating one.
Copying test-file config.yaml
```

Perform linting:
```
planemo l
Linting tool /Users/vk/Work/CHESS/ChessPipeline/chap.xml
Applying linter tests... CHECK
.. CHECK: 1 test(s) found.
Applying linter output... CHECK
.. INFO: 1 outputs found.
Applying linter inputs... CHECK
.. INFO: Found 2 input parameters.
Applying linter help... CHECK
.. CHECK: Tool contains help section.
.. CHECK: Help contains valid reStructuredText.
Applying linter general... CHECK
.. CHECK: Tool defines a version [0.1.0+galaxy0].
.. CHECK: Tool defines a name [CHESS Analysis Pipeline].
.. CHECK: Tool defines an id [chap].
.. CHECK: Tool specifies profile version [21.05].
Applying linter command... CHECK
.. INFO: Tool contains a command.
Applying linter citations... CHECK
.. CHECK: Found 1 likely valid citations.
Applying linter tool_xsd... CHECK
.. INFO: File validates against XML schema.
```

Now, we can start server via the following commad: `planemo s`,
it will take a while. Once finished we may visit
`http://127.0.0.1:9090` to see our galaxy hub along with
our pipeline tool.
