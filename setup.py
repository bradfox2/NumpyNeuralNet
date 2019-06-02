import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='NNN',  
     version='0.1',
     author="Bradley Fox",
     author_email="bradfox2@gmail.com",
     description="Basic NN for learning",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/bradfox2",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )