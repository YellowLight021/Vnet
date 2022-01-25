# import os
# os.system('pip install patoolib ')
import patoolib

archive_file="F:\LUNA16\seg-lungs-LUNA16.rar"
patoolib.extract_archive(archive_file, outdir=r"F:\LUNA16\test")
# rf = rarfile.RarFile("F:\LUNA16\seg-lungs-LUNA16.rar")
# rf.extractall(r"F:\LUNA16\test")
print('finished')