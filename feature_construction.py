import torch

class feature_library():
    """ Class object that holds the features for constructing the functions basis.

        Attributes:
            functions (list)        : list of function objects
            nx (int)                : the number of states
            nu (int)                : the number of inputs
            include_one (bool)      : choose to include an offset
            interaction_only (bool) : true exlcudes terms such as x1[k]*x2[k]

        Methods:
            __init__(self, functions, nx, nu, include_one, interaction_only):
                Constructor

            fit_transform(self, X):
                Transforms data in X using the specified functions stored in self.functions.

            feature_names(self):
                Returns a list of strings which correspond to the features in self.functions.

            feature_number(self):
                Returns the number of features that will be produced by transformation.

    """
    def __init__(
            self,
            functions,
            nx,
            nu,
            include_one = True,
            interaction_only=True
    ):
        self.functions = functions
        # TODO: add interaction only to include/exclude cross terms
        # now functions are applied to all states and inputs
        self.interaction_only = interaction_only

        # set to false to exclude possible offset
        self.include_one = include_one

        # state and input dimensions
        self.nx = nx
        self.nu = nu

        # creates list of factors that are in system
        # x0[k],..., xn[k], u0[k],..., un[k]
        self.term_list = [f"x{i}[k]" for i in range(self.nx)]
        input_list = [f"u{i}[k]" for i in range(self.nu)]
        self.term_list.extend(input_list)

    def fit_transform(self, X):
        # if include_one = True add term for offset
        out_feature = ((X[:,0])**0).unsqueeze(1) if self.include_one else torch.empty(X.shape[0], 1)
        if self.interaction_only:
            for f in self.functions:
                out_feature = torch.hstack((out_feature, f(X)[0]))
            return out_feature
        # TODO: add the stuff for cross terms
        
    def feature_names(self):
        # returns list with feature names
        flist = ["1"] if self.include_one else []
        for f in self.functions:
            for x in self.term_list:
                flist.append(f(torch.tensor(1.),f"{x}")[-1])
        return flist
    
    def feature_number(self):
        if self.interaction_only:
            fnum =  len(self.functions*(self.nx+self.nu))
        # TODO: add functionality for cross features when interaction only is false

        return fnum+self.include_one
    
def f(x, name="_"):
  """ f(x) = x """
  return x, f"{name}"

def f2(x, name="_"):
  """ f(x) = x**2 """
  return x**2, f"{name}**2"

def f3(x, name="_"):
  """ f(x) = x**3 """
  return x**3, f"{name}**3"

def sin(x, name="_"):
  """ f(x) = sin(x) """
  return torch.sin(x), f"sin({name})"