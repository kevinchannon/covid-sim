#include <cmath>
#include <stdio.h>
#include <cstddef>
#include "Kernels.h"
#include "Error.h"
#include "Dist.h"
#include "Param.h"

double(*Kernel)(double);

// To speed up calculation of kernel values we provide a couple of lookup
// tables.
//
// nKernel is a g_allParams.NKR+1 element table of lookups nKernel[0] is the kernel
// value at a distance of 0, and nKernel[g_allParams.NKR] is the kernel value at the
// largest possible distance (diagonal across the bounding box).
//
// nKernelHR is a higher-resolution table of lookups, also of g_allParams.NKR+1
// elements.  nKernelHR[n * g_allParams.NK_HR] corresponds to nKernel[n] for
// n in [0, g_allParams.NKR/g_allParams.NK_HR]
//
// Graphically:
//
// Distance 0            ...                              Bound Box diagonal
//          nKernel[0]   ... nKernel[g_allParams.NKR / g_allParams.NK_HR] ... nKernel[g_allParams.NKR]
//          nKernelHR[0] ... nKernelHR[g_allParams.NKR]
double *nKernel, *nKernelHR;
void InitKernel(int DoPlaces, double norm)
{
	int i, j;

	if (g_allParams.KernelType == 1)
		Kernel = ExpKernel;
	else if (g_allParams.KernelType == 2)
		Kernel = PowerKernel;
	else if (g_allParams.KernelType == 3)
		Kernel = GaussianKernel;
	else if (g_allParams.KernelType == 4)
		Kernel = StepKernel;
	else if (g_allParams.KernelType == 5)
		Kernel = PowerKernelB;
	else if (g_allParams.KernelType == 6)
		Kernel = PowerKernelUS;
	else if (g_allParams.KernelType == 7)
		Kernel = PowerExpKernel;
#pragma omp parallel for private(i) schedule(static,500) //added private i
	for (i = 0; i <= g_allParams.kernelLookupTableSize; i++)
	{
		nKernel[i] = (*Kernel)(((double)i) * g_allParams.KernelDelta) / norm;
		nKernelHR[i] = (*Kernel)(((double)i) * g_allParams.KernelDelta / g_allParams.hiResKernelExpansionFactor) / norm;
	}

#pragma omp parallel for schedule(static,500) private(i,j)
	for (i = 0; i < g_allParams.populatedCellCount; i++)
	{
		cell *l = CellLookup[i];
		l->tot_prob = 0;
		for (j = 0; j < g_allParams.populatedCellCount; j++)
		{
			cell *m = CellLookup[j];
			l->tot_prob += (l->max_trans[j] = (float)numKernel(dist2_cc_min(l, m))) * m->n;
		}
	}
}

//// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// ****
//// **** KERNEL DEFINITIONS

double ExpKernel(double r2)
{
	return exp(-sqrt(r2) / g_allParams.KernelScale);
}
double PowerKernel(double r2)
{
	double t;

	t = -g_allParams.KernelShape * log(sqrt(r2) / g_allParams.KernelScale + 1);

	return (t < -690) ? 0 : exp(t);
}
double PowerKernelB(double r2)
{
	double t;

	t = 0.5 * g_allParams.KernelShape * log(r2 / (g_allParams.KernelScale * g_allParams.KernelScale));

	return (t > 690) ? 0 : (1 / (exp(t) + 1));
}
double PowerKernelUS(double r2)
{
	double t;

	t = log(sqrt(r2) / g_allParams.KernelScale + 1);

	return (t < -690) ? 0 : (exp(-g_allParams.KernelShape * t) + g_allParams.KernelP3 * exp(-g_allParams.KernelP4 * t)) / (1 + g_allParams.KernelP3);
}
double GaussianKernel(double r2)
{
	return exp(-r2 / (g_allParams.KernelScale * g_allParams.KernelScale));
}
double StepKernel(double r2)
{
	return (r2 > g_allParams.KernelScale * g_allParams.KernelScale) ? 0 : 1;
}
double PowerExpKernel(double r2)
{
	double d, t;

	d = sqrt(r2);
	t = -g_allParams.KernelShape * log(d / g_allParams.KernelScale + 1);

	return (t < -690) ? 0 : exp(t - pow(d / g_allParams.KernelP3, g_allParams.KernelP4));
}
double numKernel(double r2)
{
	double t, s;

	t = r2 / g_allParams.KernelDelta;
	if (t > g_allParams.kernelLookupTableSize)
	{
		fprintf(stderr, "** %lg  %lg  %lg**\n", r2, g_allParams.KernelDelta, t);
		ERR_CRITICAL("r too large in NumKernel\n");
	}
	s = t * g_allParams.hiResKernelExpansionFactor;
	if (s < g_allParams.kernelLookupTableSize)
	{
		t = s - floor(s);
		t = (1 - t) * nKernelHR[(int)s] + t * nKernelHR[(int)(s + 1)];
	}
	else
	{
		s = t - floor(t);
		t = (1 - s) * nKernel[(int)t] + s * nKernel[(int)(t + 1)];
	}
	return t;
}
