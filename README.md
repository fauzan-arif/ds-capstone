# DS Capstone Project without a name

## Setup

Use the requirements file in this repo to create a new environment.

```BASH
make
```

Don't forget to activate the enviroment
```BASH
source .venv/bin/activate
```

### Other make commands

create the `.venv` directory
```BASH
make venv
```

deletes the `.venv` directory
```BASH
make clean 
```

upgrades pip
```BASH
make pip 
```

install pip requirements.txt
```BASH
make requirements 
```