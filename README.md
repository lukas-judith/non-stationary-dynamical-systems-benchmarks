# Non-Stationary Dynamical Systems Benchmarks

This code was created for my master's thesis project at Heidelberg University, in the [group of Prof. Daniel Durstewitz](https://durstewitzlab.github.io). The focus of my thesis is the reconstruction of dynamical systems with non-stationary parameters from time series data, using recurrent neural networks. The code provided can be used to generate different benchmark systems to assess the performance of such models.



**How to use the code:**

*Prerequisites: [conda](https://docs.conda.io/en/latest/) must be installed.*

1. Create a new conda environment from the .yml file, using 

   ```
   conda env create --file env.yml
   ```

2. Activate the environment:

   ```
   conda activate ds_gen
   ```

3. Modify the generation settings in the generate_datasets.py file (some pre-defined example settings are provided in the file)

4. To generate a new dataset, run the command

   ```
   python generate_datasets.py
   ```



**TODO:**

- Add code for reading the generation settings from a JSON file
