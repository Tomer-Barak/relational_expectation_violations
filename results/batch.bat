@echo off
setlocal enabledelayedexpansion

set "old_string=repeated_experiments"
set "new_string=res"

for /d %%f in (%old_string%*) do (
    set "old_name=%%f"
    set "new_name=!old_name:%old_string%=%new_string%!"
    ren "%%f" "!new_name!"
)

endlocal
