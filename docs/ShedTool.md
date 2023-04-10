## ShedTool

This document provides full set of instructions how to place a package
into ShedTool in galaxy server.

### Install and/or activate planemo
Based on this instructions: /home/rv43/planemo.readme

Activate environment:
```
. planemo/bin/activate
```

Install planemo:
```
python3 -m venv planemo
. planemo/bin/activate
pip install planemo
```
As needed:
```
pip install --upgrade pip
pip install -U planemo (to update planemo)
```

Next, we need to create `.shed.yml` file
Create a `.shed.yml` with the right credentials and
make the name in there the same as in the repo,
see [instructions here](https://galaxy-iuc-standards.readthedocs.io/en/latest/best_practices/shed_yml.html). Please note:
- for type use: unrestricted
- and choose one of the categories this [repo](https://testtoolshed.g2.bx.psu.edu/repository)

### Registion step
To use our shed tool we need to register at
https://testtoolshed.g2.bx.psu.edu
and get the API key.

### Shed tool
Create a new tool with planemo:
- create your shed tool xml file, e.g. chap.xml
- `planemo shed_create --shed_target testtoolshed --shed_key <API key>`
get the API key from user prefferences in the testtoolshed
- or update existing shed tool:
```
planemo shed_update --check_diff --shed_target testtoolshed --shed_key <API key>
```

Optionally, we may check a tool XML file for common problems:
````
planemo lint [tool.xml]
planemo test [tool.xml]
```
Check https://testtoolshed.g2.bx.psu.edu/ for your tool, e.g.
search for your tool name and you should see:
```
Tool id and version      Repository name↓           Owner       Revision
chapmlaas: 0.1.0+galaxy0 package_chapmlaas_0_0_2    vkuznet     0:5ec75e142574 (2023-04-05)
```

### Upload tool to galaxy-dev
To install the tool on galaxy-dev:
1. Login as an administrator to https://galaxy-dev.classe.cornell.edu
2. Go to the "Admin" tab at the top of the page
3. In the sidebar on the left, go to "Install and uninstall" (admin->Tool
Management->Install and Uninstall)
4. Search for your testtoolshed tool
5. Click "Install"
6. Select a target section (e.g. X-IMG) and click "OK"

### Manual procedure
If the testtoolshed does not show up, in config/galaxy.yml add:
```
    tool_sheds_config_file: /nfs/chess/galaxy-dev/server/config/tool_sheds_conf.xml
```
then create:
```
/nfs/chess/galaxy-dev/server/config/tool_sheds_conf.xml
```
with:
```
<?xml version="1.0"?>
<tool_sheds>
    <tool_shed name="Galaxy Main Tool Shed" url="https://toolshed.g2.bx.psu.edu/"/>
    <!-- Test Tool Shed should be used only for testing purposes. -->
    <tool_shed name="Galaxy Test Tool Shed" url="https://testtoolshed.g2.bx.psu.edu/"/>
</tool_sheds>
```
