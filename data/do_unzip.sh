#！/bin/bash
zip -s 0 myfile.zip --out file.zip
zip -F file.zip --out file-large.zip
unzip file-large.zip

