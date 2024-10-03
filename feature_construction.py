import pysindy as ps
import numpy as np
import torch

import polynomial as p
import fourier as f

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
            include_one=True,
            include_interataction=True,
            exclude_idx=None,
            POLY=[p.f, p.f2, p.f3, p.f4, p.f5, p.f6, p.f7, p.f8, p.f9],
            FOURIER=[f.sin, f.cos, f.sin2x, f.cos2x, f.sin3x, f.cos3x, f.sin4x, f.cos4x, f.sin5x, f.cos5x, f.sin6x, f.cos6x],
            T = np.array([1]),
            U = np.array([1])
    ):
        self.T = np.diag(T) if T.shape[0] != 1 else T[0]
        self.U = np.diag(U) if U.shape[0] != 1 else U[0]

        # TODO: add interaction only to include/exclude cross terms
        # now functions are applied to all states and inputs
        self.include_interataction = include_interataction

        # set to false to exclude possible offset
        self.include_one = include_one

        # state and input dimensions
        self.nx = nx
        self.nu = nu

        if "poly" in functions:
            self.ext_library = ps.PolynomialLibrary(degree=functions[-1], include_interaction=self.include_interataction).fit(np.random.rand(self.nx+self.nu))
        elif "fourier" in functions:
            functions = FOURIER[:2*functions[-1]]
        self.functions = functions

        # creates list of factors that are in system
        # x0[k],..., xn[k], u0[k],..., un[k]
        self.term_list = [f"x{i}[k]" for i in range(self.nx)]
        input_list = [f"u{i}[k]" for i in range(self.nu)]
        self.term_list.extend(input_list)

        # after running for new run exlude an index
        self.exclude_idx = exclude_idx
        if self.exclude_idx is not None:
            if not self.include_one and 0 not in self.exclude_idx:
                self.exclude_idx = [self.exclude_idx, 0]
            else:
                self.exclude_idx = self.exclude_idx
                    
        self.set_feature_names()

    def fit_transform(self, X):
        if "poly" not in self.functions:
            # if include_one = True add term for offset
            out_feature = ((X[:,0])**0).unsqueeze(1)
            for f in self.functions:
                # dummy call
                _, requires_T, _ = f(torch.tensor(1.), 1.)

                for idx, x in enumerate(self.term_list):
                    x_col = X[:,idx]
                    if requires_T:
                        idx = int(x[1])
                        if "x" in x:
                            scaling_factor = self.T[idx]
                        elif "u" in x:
                            scaling_factor = self.U[idx]
                        new_feature, _, f_str  = f(x_col, scaling_factor)
                    else:
                        # input term doesnt require scaling factor
                        new_feature, _, f_str = f(x_col, 1.0)
                    out_feature = torch.hstack((out_feature, new_feature.unsqueeze(1)))
        else:
            out_feature = torch.tensor(self.ext_library.transform(X.detach()), requires_grad=True)

        if self.exclude_idx is not None:
            out_feature = torch.cat([out_feature[:, i].unsqueeze(1) for i in range(out_feature.size(1)) if i not in self.exclude_idx], dim=1)

        return out_feature if self.include_one else out_feature[:,:-1]
        

    def feature_names(self):
        return self.feature_names
        
    def set_feature_names(self):
        if "poly" not in self.functions:
            # returns list with feature names
            flist = ["1"] if self.include_one else []
            for f in self.functions:
                for x in self.term_list:
                    _, requires_T, func_str = f(torch.tensor(1.),self.T[0],f"{x}")
                    if requires_T:
                        idx = int(x[1])
                        if "x" in x:
                            _, _, func_str = f(torch.tensor(1.),self.T[idx],f"T{idx}*{x}")
                        elif "u" in x:
                            _, _, func_str = f(torch.tensor(1.),self.U[idx],f"U{idx}*{x}")
                        flist.append(func_str)
                    else:
                        flist.append(func_str)
        else:
            flist = self.ext_library.get_feature_names()
            for idx, val in enumerate(flist):
                val = re.sub(r'x(\d+)', r'x\1[k]', val)
                for xi in range(self.nx):
                    val = re.sub(rf'x{xi}', rf'x{xi}', val)
                for ui in range(self.nu):
                    val = val.replace(f'x{self.nx + ui}[k]', f'u{ui}[k]')
                flist[idx] = val.replace('^', '**')

        if self.exclude_idx is not None:
                self.exclude_idx.sort(reverse=True)

                for i in self.exclude_idx:
                    flist.pop(i)

        self.feature_names = flist
        return

    def feature_number(self):
        return len(self.feature_names)
    