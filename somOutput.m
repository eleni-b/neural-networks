function [output] = somOutput(pattern)
global  IW ;

NegEuc=negdist(IW,pattern);
output=compet(NegEuc);

