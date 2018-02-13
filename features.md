### Classification

The [auditNG](https://github.com/auditNG/auditNG) tool provides with a list of features and parameters which we can use for classification.

Following are the preliminary features that will be used for initial analysis:

 1. **syscall** - There are total of 326 syscalls, out of which we need to identify and flag the syscalls that are related to file read and writes.
 2. **exit** - The exit status code for the process. If it isn't a success then we need to flag it as unauthorized access, and the ones that are to look for *EACCES*(13) and *EPERM*(1). The complete list [Linux System Error Codes](http://www-numi.fnal.gov/offline_software/srt_public_context/WebDocs/Errors/unix_system_errors.html)
 3. **EXE** - the class of binary that is being run. The one to look for are which modify sensitive files and specially editors. Eg. if there is a exit code 13 for syscall fopen by an editor on a sensitive file that is not meant to be read by any program other than a specific binary, that needs to be flagged.
 4. **user** - The type of uses based on the uid number. uid < 1000 (privileged user) and uid > 1000 (unprivileged user)
 5. **filename** - The file type that is being accessed based on either it is a system file (meant to be read explicitly or not) and user file.
