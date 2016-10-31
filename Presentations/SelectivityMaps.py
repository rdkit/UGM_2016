# $Id$
#
#  Copyright (c) 2016, Novartis Institutes for BioMedical Research Inc.
#  All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: 
#
#     * Redistributions of source code must retain the above copyright 
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following 
#       disclaimer in the documentation and/or other materials provided 
#       with the distribution.
#     * Neither the name of Novartis Institutes for BioMedical Research Inc. 
#       nor the names of its contributors may be used to endorse or promote 
#       products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Based on SimilarityMaps.py by Sereina Riniker August 2013
#
# Created by Simon Ruedisser, September 2016

import math
import numpy
import matplotlib.pyplot as plt
from matplotlib import cm
from rdkit.Chem import Draw

def getProbaprod(fp, predictionFunction, target_nr = 0): 
    """probability for selectivity on given target_nr: proba for active on target multiplied with proba for inactive on other targets
    sum of predict_log_proba to avoid underflow errors for small probabilities
    """
    p = predictionFunction(fp)
    c = [0, 0, 0]
    c[target_nr] = 1
    s = p[0][0][c[0]] + p[1][0][c[1]] + p[2][0][c[2]]
    return math.exp(s)

def GetAtomicWeightsForModel(probeMol, fpFunction, predictionFunction):
    """
    Calculates the atomic weights for the probe molecule based on 
    a fingerprint function and the prediction function of a ML model.

    Parameters:
    probeMol -- the probe molecule
    fpFunction -- the fingerprint function
    predictionFunction -- the prediction function of the ML model
    """
    if hasattr(probeMol, '_fpInfo'): delattr(probeMol, '_fpInfo')
    probeFP = fpFunction(probeMol, -1)
    baseProba = predictionFunction(probeFP)
    # loop over atoms
    weights = []
    for atomId in range(probeMol.GetNumAtoms()):
        newFP = fpFunction(probeMol, atomId)
        newProba = predictionFunction(newFP)
        weights.append(baseProba - newProba)
    if hasattr(probeMol, '_fpInfo'): delattr(probeMol, '_fpInfo')
    return weights

def GetStandardizedWeights(weights, weightsScaling=True):
    """
    Normalizes the weights,
    such that the absolute maximum weight equals 1.0.

    Parameters:
    weights -- the list with the atomic weights
	weightsScaling=False do not normalize weights
    """
    tmp = [math.fabs(w) for w in weights]
    currentMax = max(tmp)         
    if ((currentMax > 0) & (weightsScaling)):
        return [w/currentMax for w in weights], currentMax
    else:
        return weights, currentMax


def GetSimilarityMapFromWeights(mol, weights, weightsScaling=True, colorMap=cm.PiYG, scale=-1, size=(250, 250), sigma=None,  #@UndefinedVariable  #pylint: disable=E1101
                                coordScale=1.5, step=0.01, colors='k', contourLines=10, alpha=0.5,  **kwargs):
    """
    Generates the similarity map for a molecule given the atomic weights.

    Parameters:
    mol -- the molecule of interest
    colorMap -- the matplotlib color map scheme
    scale -- the scaling: scale < 0 -> the absolute maximum weight is used as maximum scale
                          scale = double -> this is the maximum scale
    size -- the size of the figure
    sigma -- the sigma for the Gaussians
    coordScale -- scaling factor for the coordinates
    step -- the step for calcAtomGaussian
    colors -- color of the contour lines
    contourLines -- if integer number N: N contour lines are drawn
                    if list(numbers): contour lines at these numbers are drawn
    alpha -- the alpha blending value for the contour lines
    kwargs -- additional arguments for drawing
    """
    if mol.GetNumAtoms() < 2: raise ValueError("too few atoms")
    fig = Draw.MolToMPL(mol, coordScale=coordScale, size=size, **kwargs)
    if sigma is None:
        if mol.GetNumBonds() > 0:
            bond = mol.GetBondWithIdx(0)
            idx1 = bond.GetBeginAtomIdx()
            idx2 = bond.GetEndAtomIdx()
            sigma = 0.3 * math.sqrt(sum([(mol._atomPs[idx1][i]-mol._atomPs[idx2][i])**2 for i in range(2)]))
        else:
            sigma = 0.3 * math.sqrt(sum([(mol._atomPs[0][i]-mol._atomPs[1][i])**2 for i in range(2)]))
    sigma = round(sigma, 2)
    x, y, z = Draw.calcAtomGaussians(mol, sigma, weights=weights, step=step)
    # scaling
    if scale <= 0.0: maxScale = max(math.fabs(numpy.min(z)), math.fabs(numpy.max(z)))
    else: maxScale = scale
    # coloring
    if math.fabs(maxScale) < 1:
        maxScale = 1
    fig.axes[0].imshow(z, cmap=colorMap, interpolation='bilinear', origin='lower', extent=(0,1,0,1), vmin=-maxScale, vmax=maxScale)
    # contour lines
    # only draw lines if at least one weight is not zero
    if len([w for w in weights if w != 0.0]):
        fig.axes[0].contour(x, y, z, contourLines, colors=colors, alpha=alpha, **kwargs)
    return fig

def GetSimilarityMapForModel(probeMol, fpFunction, predictionFunction, weightsScaling=True, **kwargs):
    """
    Generates the similarity map for a given ML model and probe molecule, 
    and fingerprint function.

    Parameters:
    probeMol -- the probe molecule
    fpFunction -- the fingerprint function
    predictionFunction -- the prediction function of the ML model
    kwargs -- additional arguments for drawing
    """
    weights = GetAtomicWeightsForModel(probeMol, fpFunction, predictionFunction)
    weights, maxWeight = GetStandardizedWeights(weights, weightsScaling)
    fig = GetSimilarityMapFromWeights(probeMol, weights, **kwargs)
    return fig, maxWeight 
