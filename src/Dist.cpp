#include <stdlib.h>
#include <math.h>

#include "Constants.h"
#include "Dist.h"
#include "Param.h"

double sinx[361], cosx[361], asin2sqx[1001];

//// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** 
//// **** DISTANCE FUNCTIONS (return distance-squared, which is input for every Kernel function)
//// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** 

double dist2UTM(double x1, double y1, double x2, double y2)
{
	double x, y, cy1, cy2, yt, xi, yi;

	x = fabs(x1 - x2) / 2;
	y = fabs(y1 - y2) / 2;
	xi = floor(x);
	yi = floor(y);
	x -= xi;
	y -= yi;
	x = (1 - x) * sinx[(int)xi] + x * sinx[((int)xi) + 1];
	x = x * x;
	y = (1 - y) * sinx[(int)yi] + y * sinx[((int)yi) + 1];
	y = y * y;
	yt = fabs(y1 + g_allParams.SpatialBoundingBox[1]);
	yi = floor(yt);
	cy1 = yt - yi;
	cy1 = (1 - cy1) * cosx[((int)yi)] + cy1 * cosx[((int)yi) + 1];
	yt = fabs(y2 + g_allParams.SpatialBoundingBox[1]);
	yi = floor(yt);
	cy2 = yt - yi;
	cy2 = (1 - cy2) * cosx[((int)yi)] + cy2 * cosx[((int)yi) + 1];
	x = fabs(1000 * (y + x * cy1 * cy2));
	xi = floor(x);
	x -= xi;
	y = (1 - x) * asin2sqx[((int)xi)] + x * asin2sqx[((int)xi) + 1];
	return 4 * EARTHRADIUS * EARTHRADIUS * y;
}
double dist2(person* a, person* b)
{
	double x, y;

	if (g_allParams.DoUTM_coords)
		return dist2UTM(Households[a->hh].loc_x, Households[a->hh].loc_y, Households[b->hh].loc_x, Households[b->hh].loc_y);
	else
	{
		x = fabs(Households[a->hh].loc_x - Households[b->hh].loc_x);
		y = fabs(Households[a->hh].loc_y - Households[b->hh].loc_y);
		if (g_allParams.DoPeriodicBoundaries)
		{
			if (x > g_allParams.width * 0.5) x = g_allParams.width - x;
			if (y > g_allParams.height * 0.5) y = g_allParams.height - y;
		}
		return x * x + y * y;
	}
}
double dist2_cc(cell* a, cell* b)
{
	double x, y;
	int l, m;

	l = (int)(a - Cells);
	m = (int)(b - Cells);
	if (g_allParams.DoUTM_coords)
		return dist2UTM(g_allParams.cwidth * fabs((double)(l / g_allParams.nch)), g_allParams.cheight * fabs((double)(l % g_allParams.nch)),
			g_allParams.cwidth * fabs((double)(m / g_allParams.nch)), g_allParams.cheight * fabs((double)(m % g_allParams.nch)));
	else
	{
		x = g_allParams.cwidth * fabs((double)(l / g_allParams.nch - m / g_allParams.nch));
		y = g_allParams.cheight * fabs((double)(l % g_allParams.nch - m % g_allParams.nch));
		if (g_allParams.DoPeriodicBoundaries)
		{
			if (x > g_allParams.width * 0.5) x = g_allParams.width - x;
			if (y > g_allParams.height * 0.5) y = g_allParams.height - y;
		}
		return x * x + y * y;
	}
}
double dist2_cc_min(cell* a, cell* b)
{
	double x, y;
	int l, m, i, j;

	l = (int)(a - Cells);
	m = (int)(b - Cells);
	i = l; j = m;
	if (g_allParams.DoUTM_coords)
	{
		if (g_allParams.cwidth * ((double)abs(m / g_allParams.nch - l / g_allParams.nch)) > PI)
		{
			if (m / g_allParams.nch > l / g_allParams.nch)
				j += g_allParams.nch;
			else if (m / g_allParams.nch < l / g_allParams.nch)
				i += g_allParams.nch;
		}
		else
		{
			if (m / g_allParams.nch > l / g_allParams.nch)
				i += g_allParams.nch;
			else if (m / g_allParams.nch < l / g_allParams.nch)
				j += g_allParams.nch;
		}
		if (m % g_allParams.nch > l % g_allParams.nch)
			i++;
		else if (m % g_allParams.nch < l % g_allParams.nch)
			j++;
		return dist2UTM(g_allParams.cwidth * fabs((double)(i / g_allParams.nch)), g_allParams.cheight * fabs((double)(i % g_allParams.nch)),
			g_allParams.cwidth * fabs((double)(j / g_allParams.nch)), g_allParams.cheight * fabs((double)(j % g_allParams.nch)));
	}
	else
	{
		if ((g_allParams.DoPeriodicBoundaries) && (g_allParams.cwidth * ((double)abs(m / g_allParams.nch - l / g_allParams.nch)) > g_allParams.width * 0.5))
		{
			if (m / g_allParams.nch > l / g_allParams.nch)
				j += g_allParams.nch;
			else if (m / g_allParams.nch < l / g_allParams.nch)
				i += g_allParams.nch;
		}
		else
		{
			if (m / g_allParams.nch > l / g_allParams.nch)
				i += g_allParams.nch;
			else if (m / g_allParams.nch < l / g_allParams.nch)
				j += g_allParams.nch;
		}
		if ((g_allParams.DoPeriodicBoundaries) && (g_allParams.height * ((double)abs(m % g_allParams.nch - l % g_allParams.nch)) > g_allParams.height * 0.5))
		{
			if (m % g_allParams.nch > l % g_allParams.nch)
				j++;
			else if (m % g_allParams.nch < l % g_allParams.nch)
				i++;
		}
		else
		{
			if (m % g_allParams.nch > l % g_allParams.nch)
				i++;
			else if (m % g_allParams.nch < l % g_allParams.nch)
				j++;
		}
		x = g_allParams.cwidth * fabs((double)(i / g_allParams.nch - j / g_allParams.nch));
		y = g_allParams.cheight * fabs((double)(i % g_allParams.nch - j % g_allParams.nch));
		if (g_allParams.DoPeriodicBoundaries)
		{
			if (x > g_allParams.width * 0.5) x = g_allParams.width - x;
			if (y > g_allParams.height * 0.5) y = g_allParams.height - y;
		}
		return x * x + y * y;
	}
}
double dist2_mm(microcell* a, microcell* b)
{
	double x, y;
	int l, m;

	l = (int)(a - Mcells);
	m = (int)(b - Mcells);
	if (g_allParams.DoUTM_coords)
		return dist2UTM(g_allParams.mcwidth * fabs((double)(l / g_allParams.nmch)), g_allParams.mcheight * fabs((double)(l % g_allParams.nmch)),
			g_allParams.mcwidth * fabs((double)(m / g_allParams.nmch)), g_allParams.mcheight * fabs((double)(m % g_allParams.nmch)));
	else
	{
		x = g_allParams.mcwidth * fabs((double)(l / g_allParams.nmch - m / g_allParams.nmch));
		y = g_allParams.mcheight * fabs((double)(l % g_allParams.nmch - m % g_allParams.nmch));
		if (g_allParams.DoPeriodicBoundaries)
		{
			if (x > g_allParams.width * 0.5) x = g_allParams.width - x;
			if (y > g_allParams.height * 0.5) y = g_allParams.height - y;
		}
		return x * x + y * y;
	}
}

double dist2_raw(double ax, double ay, double bx, double by)
{
	double x, y;

	if (g_allParams.DoUTM_coords)
		return dist2UTM(ax, ay, bx, by);
	else
	{
		x = fabs(ax - bx);
		y = fabs(ay - by);
		if (g_allParams.DoPeriodicBoundaries)
		{
			if (x > g_allParams.width * 0.5) x = g_allParams.width - x;
			if (y > g_allParams.height * 0.5) y = g_allParams.height - y;
		}
		return x * x + y * y;
	}
}
