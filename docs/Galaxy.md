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

### Run galaxy
The galaxy framework can be run in two different ways:
- either via `planemo s` command, or
- by cloning galaxy repository and run `run.sh` script

#### run galaxy via planemo
`planemo` provides an easy wasy to run galaxy via `planemo s` (start the
server) command. But when you invoke this command it will fetch all
necessary dependencies for galaxy and this process will take 1/2 hour.
Unfortunately, this process does not cache dependencies, and it will be
repeated upon next call.

Therefore, we recommend to run galaxy natively.

#### run galaxy natively
To run galaxy natively you must clone its repository:
```
git clone https://github.com/galaxyproject/galaxy.git
```
After that, just run `run.sh` and it will setup python virtual environment
in `$PWD/.venv`, fetch and build all dependencies. This is one time process
and after next call the galaxy server will start quickly.

Upon successfull start you will see the following:
```
Serving on http://127.0.0.1:8080
```
and can visit this URL to access your galaxy server.

#### Use conda for galaxy
The galaxy by default relies on python virtual env and will install it in
`.venv` area. But it also supports `conda` environment. To use it you need:
- install anaconda
- source anaconda env
- start galaexy as `run.sh --no-create-venv` or `run.sh --skip-venv`

The one *big* advantage of using `conda` is that it will allow
packages to install requirements if they are specified in shed xml file.

### Adding tool manually to galaxy
- perform local install of galaxy via
```
git clone clone https://github.com/galaxyproject/galaxy.git
```
- create your new area within tools
```
cd tools
mkdir chess
cd chess
```
- copy your files to your tools area, e.g.
```
cp chap.xml data.csv img.png ~/Work/CHESS/GIT/galaxy/tools/chess
```
- adjust your galaxy `config/tool_conf.xml.sample` or `config/tool_conf.xml`
file with your new tool info, e.g.
```
<section name="CHESS" id="chess">
   <tool file="chess/chap.xml" />
</section>
```
- start galaxy via `run.sh`

For more info see this
[page](https://galaxyproject.org/admin/tools/add-tool-tutorial/)

### References
1. [Installing tools into Galaxy](https://galaxyproject.org/admin/tools/add-tool-from-toolshed-tutorial/)
2. [Adding custom tool to galaxy](https://galaxyproject.org/admin/tools/add-tool-tutorial/)
3. [how to publish tool to Shed](https://galaxyproject.org/toolshed/publish-tool/)
4. [Galaxy administration](https://galaxyproject.org/admin/)
5. [Galaxy support](https://galaxyproject.org/support/)
