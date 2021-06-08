# Bayesian Reactive Sindy

This is the code to accompany the paper "Sparse bayesian inference of mass-action biochemical reaction networks using the regularized horseshoe prior".

Examples of use and experiments presented in the paper are shown in the analysis directory in notebook format.

The `RSindy` abstract base class represents a class which
1.  Build an ansatz reaction set
2.  Construct the appropriate stocihiometric matrices, mass action reaction ratees, and descriptions
3.  Pre-process observational data 
4.  Defines a method to estimate the reaction rate constants k as described in the paper

At the moment, we automatically construct the library of ansatz reactions as defined in `utils.generate_valid_reaction_basis` method.

The `RSindyRegularizedHorseshoe` implements the `RSindy` base class with the model and estimation techniques as defined in the paper.
Broadly, the `RSindyRegularizedHorseshoe.fit_non_dx` method generates a valid Stan model using the regularized horseshoe prior and the non-derivative observational model from the stored stoichiometric matrix and reaction rate vectors.  Posterior distributions are estimated using a few default settings and the entire fit is directly returned to the user for further analysis as demonstrated in the analysis notebooks.

