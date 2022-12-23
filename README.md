# Non-Stationary Dynamical Systems Benchmarks

The code was created in part of my master's thesis at Heidelberg University, working in the [group of Prof. Daniel Durstewitz](https://durstewitzlab.github.io). The focus of my thesis is the reconstruction of dynamical systems with non-stationary parameters from time series data, using recurrent neural neural networks. The code provided can be used to generate different benchmark systems to assess the performance of such models.



**How to use the code:**

1. Create a new conda environment from the .yml file, using 

   ```
   conda env create --file env.yml
   ```

2. Open JupyterLab (or, alternatively, Jupyter Notebook) by running the command

   ```
   jupyter lab [path of the repository]
   ```

3. Modify the generation settings in the generate_datasets.py file (some pre-defined example settings are provided in the file).

4. Run the command

   ```
   python generate_datasets.py
   ```

​		to generate a new dataset.



**TODO:**

- Add code for reading the generation settings from a JSON file.
