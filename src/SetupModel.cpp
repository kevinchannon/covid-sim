#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "binio.h"
#include "Error.h"
#include "Rand.h"
#include "Kernels.h"
#include "Dist.h"
#include "MachineDefines.h"
#include "Param.h"
#include "SetupModel.h"
#include "Model.h"
#include "ModelMacros.h"
#include "SharedFuncs.h"
#include "InfStat.h"
#include "Bitmap.h"

void* BinFileBuf;
bin_file* BF;
int netbuf[NUM_PLACE_TYPES * 1000000];


///// INITIALIZE / SET UP FUNCTIONS
void SetupModel(char* DensityFile, char* NetworkFile, char* SchoolFile, char* RegDemogFile)
{
	int i, j, k, l, m, i1, i2, j2, l2, m2, tn; //added tn as variable for multi-threaded loops: 28/11/14
	int age; //added age (group): ggilani 09/03/20
	unsigned int rn;
	double t, s, s2, s3, x, y, t2, t3, d, q;
	char buf[2048];
	FILE* dat;

	if (!(Xcg1 = (long*)malloc(MAX_NUM_THREADS * CACHE_LINE_SIZE * sizeof(long)))) ERR_CRITICAL("Unable to allocate ranf storage\n");
	if (!(Xcg2 = (long*)malloc(MAX_NUM_THREADS * CACHE_LINE_SIZE * sizeof(long)))) ERR_CRITICAL("Unable to allocate ranf storage\n");
	g_allParams.nextSetupSeed1 = g_allParams.setupSeed1;
	g_allParams.nextSetupSeed2 = g_allParams.setupSeed2;
	setall(&g_allParams.nextSetupSeed1, &g_allParams.nextSetupSeed2);

	g_allParams.DoBin = -1;
	if (g_allParams.DoHeteroDensity)
	{
		fprintf(stderr, "Scanning population density file\n");
		if (!(dat = fopen(DensityFile, "rb"))) ERR_CRITICAL("Unable to open density file\n");
		fread_big(&(g_allParams.binFileLen), sizeof(unsigned int), 1, dat);
		if (g_allParams.binFileLen == 0xf0f0f0f0) //code for first 4 bytes of binary file ## NOTE - SHOULD BE LONG LONG TO COPE WITH BIGGER POPULATIONS
		{
			g_allParams.DoBin = 1;
			fread_big(&(g_allParams.binFileLen), sizeof(unsigned int), 1, dat);
			if (!(BinFileBuf = (void*)malloc(g_allParams.binFileLen * sizeof(bin_file)))) ERR_CRITICAL("Unable to allocate binary file buffer\n");
			fread_big(BinFileBuf, sizeof(bin_file), (size_t)g_allParams.binFileLen, dat);
			BF = (bin_file*)BinFileBuf;
			fclose(dat);
		}
		else
		{
			g_allParams.DoBin = 0;
			// Count the number of lines in the density file
			rewind(dat);
			g_allParams.binFileLen = 0;
			while(fgets(buf, sizeof(buf), dat) != NULL) g_allParams.binFileLen++;
			if(ferror(dat)) ERR_CRITICAL("Error while reading density file\n");
			// Read each line, and build the binary structure that corresponds to it
			rewind(dat);
			if (!(BinFileBuf = (void*)malloc(g_allParams.binFileLen * sizeof(bin_file)))) ERR_CRITICAL("Unable to allocate binary file buffer\n");
			BF = (bin_file*)BinFileBuf;
			int index = 0;
			while(fgets(buf, sizeof(buf), dat) != NULL)
			{
				// This shouldn't be able to happen, as we just counted the number of lines:
				if (index == g_allParams.binFileLen) ERR_CRITICAL("Too many input lines while reading density file\n");
				if (g_allParams.DoAdUnits)
				{
					sscanf(buf, "%lg %lg %lg %i %i", &x, &y, &t, &i2, &l);
					if (l / g_allParams.CountryDivisor != i2)
					{
						//fprintf(stderr,"# %lg %lg %lg %i %i\n",x,y,t,i2,l);
					}
				}
				else {
					sscanf(buf, "%lg %lg %lg %i", &x, &y, &t, &i2);
					l = 0;
				}
				BF[index].x = x;
				BF[index].y = y;
				BF[index].pop = t;
				BF[index].cnt = i2;
				BF[index].ad = l;
				index++;
			}
			if(ferror(dat)) ERR_CRITICAL("Error while reading density file\n");
			// This shouldn't be able to happen, as we just counted the number of lines:
			if (index != g_allParams.binFileLen) ERR_CRITICAL("Too few input lines while reading density file\n");
			fclose(dat);
		}

		if (g_allParams.DoAdunitBoundaries)
		{
			// We will compute a precise spatial bounding box using the population locations.
			// Initially, set the min values too high, and the max values too low, and then
			// we will adjust them as we read population data.
			g_allParams.SpatialBoundingBox[0] = g_allParams.SpatialBoundingBox[1] = 1e10;
			g_allParams.SpatialBoundingBox[2] = g_allParams.SpatialBoundingBox[3] = -1e10;
			s2 = 0;
			for (rn = 0; rn < g_allParams.binFileLen; rn++)
			{
				x = BF[rn].x;
				y = BF[rn].y;
				t = BF[rn].pop;
				i2 = BF[rn].cnt;
				l = BF[rn].ad;
				//					fprintf(stderr,"# %lg %lg %lg %i\t",x,y,t,l);

				m = (l % g_allParams.AdunitLevel1Mask) / g_allParams.AdunitLevel1Divisor;
				if (g_allParams.AdunitLevel1Lookup[m] >= 0)
					if (AdUnits[g_allParams.AdunitLevel1Lookup[m]].id / g_allParams.AdunitLevel1Mask == l / g_allParams.AdunitLevel1Mask)
					{
						AdUnits[g_allParams.AdunitLevel1Lookup[m]].cnt_id = i2;
						s2 += t;
						// Adjust the bounds of the spatial bounding box so that they include the location
						// for this block of population.
						if (x < g_allParams.SpatialBoundingBox[0]) g_allParams.SpatialBoundingBox[0] = x;
						if (x >= g_allParams.SpatialBoundingBox[2]) g_allParams.SpatialBoundingBox[2] = x + 1e-6;
						if (y < g_allParams.SpatialBoundingBox[1]) g_allParams.SpatialBoundingBox[1] = y;
						if (y >= g_allParams.SpatialBoundingBox[3]) g_allParams.SpatialBoundingBox[3] = y + 1e-6;
					}
			}
			if (!g_allParams.DoSpecifyPop) g_allParams.populationSize = (int)s2;
		}

		g_allParams.cheight = g_allParams.cwidth;
		g_allParams.SpatialBoundingBox[0] = floor(g_allParams.SpatialBoundingBox[0] / g_allParams.cwidth) * g_allParams.cwidth;
		g_allParams.SpatialBoundingBox[1] = floor(g_allParams.SpatialBoundingBox[1] / g_allParams.cheight) * g_allParams.cheight;
		g_allParams.SpatialBoundingBox[2] = ceil(g_allParams.SpatialBoundingBox[2] / g_allParams.cwidth) * g_allParams.cwidth;
		g_allParams.SpatialBoundingBox[3] = ceil(g_allParams.SpatialBoundingBox[3] / g_allParams.cheight) * g_allParams.cheight;
		g_allParams.width = g_allParams.SpatialBoundingBox[2] - g_allParams.SpatialBoundingBox[0];
		g_allParams.height = g_allParams.SpatialBoundingBox[3] - g_allParams.SpatialBoundingBox[1];
		g_allParams.ncw = 4 * ((int)ceil(g_allParams.width / g_allParams.cwidth / 4));
		g_allParams.nch = 4 * ((int)ceil(g_allParams.height / g_allParams.cheight / 4));
		g_allParams.width = ((double)g_allParams.ncw) * g_allParams.cwidth;
		g_allParams.height = ((double)g_allParams.nch) * g_allParams.cheight;
		g_allParams.SpatialBoundingBox[2] = g_allParams.SpatialBoundingBox[0] + g_allParams.width;
		g_allParams.SpatialBoundingBox[3] = g_allParams.SpatialBoundingBox[1] + g_allParams.height;
		g_allParams.cellCount = g_allParams.ncw * g_allParams.nch;
		fprintf(stderr, "Adjusted bounding box = (%lg, %lg)- (%lg, %lg)\n", g_allParams.SpatialBoundingBox[0], g_allParams.SpatialBoundingBox[1], g_allParams.SpatialBoundingBox[2], g_allParams.SpatialBoundingBox[3]);
		fprintf(stderr, "Number of cells = %i (%i x %i)\n", g_allParams.cellCount, g_allParams.ncw, g_allParams.nch);
		fprintf(stderr, "Population size = %i \n", g_allParams.populationSize);
		s = 1;
		g_allParams.DoPeriodicBoundaries = 0;
	}
	else
	{
		g_allParams.ncw = g_allParams.nch = (int)sqrt((double)g_allParams.cellCount);
		g_allParams.cellCount = g_allParams.ncw * g_allParams.nch;
		fprintf(stderr, "Number of cells adjusted to be %i (%i^2)\n", g_allParams.cellCount, g_allParams.ncw);
		s = floor(sqrt((double)g_allParams.populationSize));
		g_allParams.SpatialBoundingBox[0] = g_allParams.SpatialBoundingBox[1] = 0;
		g_allParams.SpatialBoundingBox[2] = g_allParams.SpatialBoundingBox[3] = s;
		g_allParams.populationSize = (int)(s * s);
		fprintf(stderr, "Population size adjusted to be %i (%lg^2)\n", g_allParams.populationSize, s);
		g_allParams.width = g_allParams.height = s;
		g_allParams.cwidth = g_allParams.width / ((double)g_allParams.ncw);
		g_allParams.cheight = g_allParams.height / ((double)g_allParams.nch);
	}
	g_allParams.microcellCount = g_allParams.microcellsOnACellSide * g_allParams.microcellsOnACellSide * g_allParams.cellCount;
	g_allParams.nmcw = g_allParams.ncw * g_allParams.microcellsOnACellSide;
	g_allParams.nmch = g_allParams.nch * g_allParams.microcellsOnACellSide;
	fprintf(stderr, "Number of micro-cells = %i\n", g_allParams.microcellCount);
	g_allParams.scalex = g_allParams.BitmapScale;
	g_allParams.scaley = g_allParams.BitmapAspectScale * g_allParams.BitmapScale;
	g_allParams.bwidth = (int)(g_allParams.width * (g_allParams.BoundingBox[2] - g_allParams.BoundingBox[0]) * g_allParams.scalex);
	g_allParams.bwidth = (g_allParams.bwidth + 3) / 4;
	g_allParams.bwidth *= 4;
	g_allParams.bheight = (int)(g_allParams.height * (g_allParams.BoundingBox[3] - g_allParams.BoundingBox[1]) * g_allParams.scaley);
	g_allParams.bheight += (4 - g_allParams.bheight % 4) % 4;
	g_allParams.bheight2 = g_allParams.bheight + 20; // space for colour legend
	fprintf(stderr, "Bitmap width = %i\nBitmap height = %i\n", g_allParams.bwidth, g_allParams.bheight);
	g_allParams.bminx = (int)(g_allParams.width * g_allParams.BoundingBox[0] * g_allParams.scalex);
	g_allParams.bminy = (int)(g_allParams.height * g_allParams.BoundingBox[1] * g_allParams.scaley);
	g_allParams.mcwidth = g_allParams.cwidth / ((double)g_allParams.microcellsOnACellSide);
	g_allParams.mcheight = g_allParams.cheight / ((double)g_allParams.microcellsOnACellSide);
	for (i = 0; i < g_allParams.NumSeedLocations; i++)
	{
		g_allParams.LocationInitialInfection[i][0] -= g_allParams.SpatialBoundingBox[0];
		g_allParams.LocationInitialInfection[i][1] -= g_allParams.SpatialBoundingBox[1];
	}
	t = dist2_raw(0, 0, g_allParams.width, g_allParams.height);
	if (g_allParams.DoPeriodicBoundaries) t *= 0.25;
	if (!(nKernel = (double*)calloc(g_allParams.kernelLookupTableSize + 1, sizeof(double)))) ERR_CRITICAL("Unable to allocate kernel storage\n");
	if (!(nKernelHR = (double*)calloc(g_allParams.kernelLookupTableSize + 1, sizeof(double)))) ERR_CRITICAL("Unable to allocate kernel storage\n");
	g_allParams.KernelDelta = t / g_allParams.kernelLookupTableSize;
	//	fprintf(stderr,"** %i %lg %lg %lg %lg | %lg %lg %lg %lg \n",P.DoUTM_coords,P.SpatialBoundingBox[0],P.SpatialBoundingBox[1],P.SpatialBoundingBox[2],P.SpatialBoundingBox[3],P.width,P.height,t,P.KernelDelta);
	fprintf(stderr, "Coords xmcell=%lg m   ymcell = %lg m\n", sqrt(dist2_raw(g_allParams.width / 2, g_allParams.height / 2, g_allParams.width / 2 + g_allParams.mcwidth, g_allParams.height / 2)), sqrt(dist2_raw(g_allParams.width / 2, g_allParams.height / 2, g_allParams.width / 2, g_allParams.height / 2 + g_allParams.mcheight)));
	g_allParams.KernelShape = g_allParams.MoveKernelShape;
	g_allParams.KernelScale = g_allParams.MoveKernelScale;
	g_allParams.KernelP3 = g_allParams.MoveKernelP3;
	g_allParams.KernelP4 = g_allParams.MoveKernelP4;
	g_allParams.KernelType = g_allParams.moveKernelType;
	t2 = 0.0;

	SetupPopulation(DensityFile, SchoolFile, RegDemogFile);
	if (!(TimeSeries = (results*)calloc(g_allParams.totalSampleNumber, sizeof(results)))) ERR_CRITICAL("Unable to allocate results storage\n");
	if (!(TSMeanE = (results*)calloc(g_allParams.totalSampleNumber, sizeof(results)))) ERR_CRITICAL("Unable to allocate results storage\n");
	if (!(TSVarE = (results*)calloc(g_allParams.totalSampleNumber, sizeof(results)))) ERR_CRITICAL("Unable to allocate results storage\n");
	if (!(TSMeanNE = (results*)calloc(g_allParams.totalSampleNumber, sizeof(results)))) ERR_CRITICAL("Unable to allocate results storage\n");
	if (!(TSVarNE = (results*)calloc(g_allParams.totalSampleNumber, sizeof(results)))) ERR_CRITICAL("Unable to allocate results storage\n");
	TSMean = TSMeanE; TSVar = TSVarE;

	///// This loops over index l twice just to reset the pointer TSMean from TSMeanE to TSMeanNE (same for TSVar).
	for (l = 0; l < 2; l++)
	{
		for (i = 0; i < g_allParams.totalSampleNumber; i++)
		{
			TSMean[i].S = TSMean[i].I = TSMean[i].R = TSMean[i].D = TSMean[i].L =
				TSMean[i].incL = TSMean[i].incI = TSMean[i].incR = TSMean[i].incC = TSMean[i].incDC = TSMean[i].cumDC =
				TSMean[i].incTC = TSMean[i].cumT = TSMean[i].cumTP = TSMean[i].cumUT = TSMean[i].cumV = TSMean[i].H = TSMean[i].incH =
				TSMean[i].incCT = TSMean[i].CT = TSMean[i].incCC = TSMean[i].incDCT = TSMean[i].DCT = //added contact tracing, cases who are contactsTSMean[i].cumTmax = TSMean[i].cumVmax = TSMean[i].incD = TSMean[i].incHQ = TSMean[i].incAC =
				TSMean[i].incAH = TSMean[i].incAA = TSMean[i].incACS = TSMean[i].incAPC =
				TSMean[i].incAPA = TSMean[i].incAPCS = TSMean[i].Rdenom = 0;
			TSVar[i].S = TSVar[i].I = TSVar[i].R = TSVar[i].D = TSVar[i].L =
				TSVar[i].incL = TSVar[i].incI = TSVar[i].incR = TSVar[i].incC = TSVar[i].incTC = TSVar[i].incD = TSVar[i].H = TSVar[i].incH = TSVar[i].incCT = TSVar[i].CT = TSVar[i].incCC = TSMean[i].incDCT = TSVar[i].DCT = 0;
			for (j = 0; j < NUM_PLACE_TYPES; j++) TSMean[i].PropPlacesClosed[j] = TSVar[i].PropPlacesClosed[j] = 0;
			for (j = 0; j < INFECT_TYPE_MASK; j++) TSMean[i].incItype[j] = TSMean[i].Rtype[j] = 0;
			for (j = 0; j < NUM_AGE_GROUPS; j++) TSMean[i].incCa[j] = TSMean[i].incIa[j] = TSMean[i].incDa[j] = TSMean[i].Rage[j] = 0;
			for (j = 0; j < 2; j++)
				TSMean[i].incI_keyworker[j] = TSVar[i].incI_keyworker[j] =
				TSMean[i].incC_keyworker[j] = TSVar[i].incC_keyworker[j] =
				TSMean[i].cumT_keyworker[j] = TSVar[i].cumT_keyworker[j] = 0;
			if (g_allParams.DoAdUnits)
				for (j = 0; j <= g_allParams.NumAdunits; j++)
					TSMean[i].incI_adunit[j] = TSVar[i].incI_adunit[j] =
					TSMean[i].incC_adunit[j] = TSVar[i].incC_adunit[j] =
					TSMean[i].incD_adunit[j] = TSVar[i].incD_adunit[j] =
					TSMean[i].incDC_adunit[j] = TSVar[i].incDC_adunit[j] =//added detected cases here: ggilani 03/02/15
					TSMean[i].incH_adunit[j] = TSVar[i].incH_adunit[j] =
					TSMean[i].incCT_adunit[j] = TSVar[i].incCT_adunit[j] = //added contact tracing
					TSMean[i].incCC_adunit[j] = TSVar[i].incCC_adunit[j] = //added cases who are contacts: ggilani 28/05/2019
					TSMean[i].incDCT_adunit[j] = TSVar[i].incDCT_adunit[j] = //added digital contact tracing: ggilani 11/03/20
					TSMean[i].cumT_adunit[j] = TSVar[i].cumT_adunit[j] = 0;

			if (g_allParams.DoSeverity)
			{
				//// TSMean (each severity for prevalence, incidence and cumulative incidence)
				TSMean[i].Mild = TSMean[i].ILI = TSMean[i].SARI = TSMean[i].Critical = TSMean[i].CritRecov =
					TSMean[i].incMild = TSMean[i].incILI = TSMean[i].incSARI = TSMean[i].incCritical = TSMean[i].incCritRecov =
					TSMean[i].incDeath_ILI = TSMean[i].incDeath_SARI = TSMean[i].incDeath_Critical =
					TSMean[i].cumDeath_ILI = TSMean[i].cumDeath_SARI = TSMean[i].cumDeath_Critical =
					TSMean[i].cumMild = TSMean[i].cumILI = TSMean[i].cumSARI = TSMean[i].cumCritical = TSMean[i].cumCritRecov = 0;

				//// TSVar (each severity for prevalence, incidence and cumulative incidence)
				TSVar[i].Mild = TSVar[i].ILI = TSVar[i].SARI = TSVar[i].Critical = TSVar[i].CritRecov =
					TSVar[i].incMild = TSVar[i].incILI = TSVar[i].incSARI = TSVar[i].incCritical = TSVar[i].incCritRecov =
					TSVar[i].cumMild = TSVar[i].cumILI = TSVar[i].cumSARI = TSVar[i].cumCritical = TSVar[i].cumCritRecov = 0;

				//// TSMean admin unit (each severity for prevalence, incidence and cumulative incidence by admin unit)
				if (g_allParams.DoAdUnits)
					for (j = 0; j <= g_allParams.NumAdunits; j++)
						TSMean[i].Mild_adunit[j] = TSMean[i].ILI_adunit[j] = TSMean[i].SARI_adunit[j] = TSMean[i].Critical_adunit[j] = TSMean[i].CritRecov_adunit[j] =
						TSMean[i].incMild_adunit[j] = TSMean[i].incILI_adunit[j] = TSMean[i].incSARI_adunit[j] = TSMean[i].incCritical_adunit[j] = TSMean[i].incCritRecov_adunit[j] =
						TSMean[i].incDeath_ILI_adunit[j] = TSMean[i].incDeath_SARI_adunit[j] = TSMean[i].incDeath_Critical_adunit[j] =
						TSMean[i].cumDeath_ILI_adunit[j] = TSMean[i].cumDeath_SARI_adunit[j] = TSMean[i].cumDeath_Critical_adunit[j] =
						TSMean[i].cumMild_adunit[j] = TSMean[i].cumILI_adunit[j] = TSMean[i].cumSARI_adunit[j] = TSMean[i].cumCritical_adunit[j] = TSMean[i].cumCritRecov_adunit[j] = 0;
			}
		}
		TSMean = TSMeanNE; TSVar = TSVarNE;
	}

	//added memory allocation and initialisation of infection event log, if DoRecordInfEvents is set to 1: ggilani - 10/10/2014
	if (g_allParams.DoRecordInfEvents)
	{
		if (!(InfEventLog = (events*)calloc(g_allParams.MaxInfEvents, sizeof(events)))) ERR_CRITICAL("Unable to allocate events storage\n");
		if (!(nEvents = (int*)calloc(1, sizeof(int)))) ERR_CRITICAL("Unable to allocate events storage\n");
	}

	if(g_allParams.OutputNonSeverity) SaveAgeDistrib();

	fprintf(stderr, "Initialising places...\n");
	if (g_allParams.DoPlaces)
	{
		if (g_allParams.LoadSaveNetwork == 1)
			LoadPeopleToPlaces(NetworkFile);
		else
			AssignPeopleToPlaces();
	}


	if ((g_allParams.DoPlaces) && (g_allParams.LoadSaveNetwork == 2))
		SavePeopleToPlaces(NetworkFile);
	//SaveDistribs();

	// From here on, we want the same random numbers regardless of whether we used the RNG to make the network,
	// or loaded the network from a file. Therefore we need to reseed the RNG.
	setall(&g_allParams.nextSetupSeed1, &g_allParams.nextSetupSeed2);

	StratifyPlaces();
	for (i = 0; i < g_allParams.cellCount; i++)
	{
		Cells[i].S = Cells[i].n;
		Cells[i].L = Cells[i].I = Cells[i].R = 0;
		//Cells[i].susceptible=Cells[i].members; //added this line
	}
	for (i = 0; i < g_allParams.populationSize; i++) Hosts[i].keyworker = 0;
	g_allParams.KeyWorkerNum = g_allParams.KeyWorkerIncHouseNum = m = l = 0;

	fprintf(stderr, "Initialising kernel...\n");
	InitKernel(0, 1.0);

	if (g_allParams.DoPlaces)
	{
		while ((m < g_allParams.KeyWorkerPopNum) && (l < 1000))
		{
			i = (int)(((double)g_allParams.populationSize) * ranf_mt(0));
			if (Hosts[i].keyworker)
				l++;
			else
			{
				Hosts[i].keyworker = 1;
				m++;
				g_allParams.KeyWorkerNum++;
				g_allParams.KeyWorkerIncHouseNum++;
				l = 0;
				if (ranf_mt(0) < g_allParams.KeyWorkerHouseProp)
				{
					l2 = Households[Hosts[i].hh].FirstPerson;
					m2 = l2 + Households[Hosts[i].hh].nh;
					for (j2 = l2; j2 < m2; j2++)
						if (!Hosts[j2].keyworker)
						{
							Hosts[j2].keyworker = 1;
							g_allParams.KeyWorkerIncHouseNum++;
						}
				}
			}
		}
		for (j = 0; j < g_allParams.PlaceTypeNoAirNum; j++)
		{
			m = l = 0;
			while ((m < g_allParams.KeyWorkerPlaceNum[j]) && (l < 1000))
			{
				k = (int)(((double)g_allParams.Nplace[j]) * ranf_mt(0));
				for (i2 = 0; (m < g_allParams.KeyWorkerPlaceNum[j]) && (i2 < Places[j][k].n); i2++)
				{
					i = Places[j][k].members[i2];
					if ((i < 0) || (i >= g_allParams.populationSize)) fprintf(stderr, "## %i # ", i);
					if ((Hosts[i].keyworker) || (ranf_mt(0) >= g_allParams.KeyWorkerPropInKeyPlaces[j]))
						l++;
					else
					{
						Hosts[i].keyworker = 1;
						m++;
						g_allParams.KeyWorkerNum++;
						g_allParams.KeyWorkerIncHouseNum++;
						l = 0;
						l2 = Households[Hosts[i].hh].FirstPerson;
						m2 = l2 + Households[Hosts[i].hh].nh;
						for (j2 = l2; j2 < m2; j2++)
							if ((!Hosts[j2].keyworker) && (ranf_mt(0) < g_allParams.KeyWorkerHouseProp))
							{
								Hosts[j2].keyworker = 1;
								g_allParams.KeyWorkerIncHouseNum++;
							}
					}
				}
			}
		}
		if (g_allParams.KeyWorkerNum > 0) fprintf(stderr, "%i key workers selected in total\n", g_allParams.KeyWorkerNum);
		if (g_allParams.DoAdUnits)
		{
			for (i = 0; i < g_allParams.NumAdunits; i++) AdUnits[i].NP = 0;
			for (j = 0; j < g_allParams.PlaceTypeNum; j++)
				if (g_allParams.PlaceCloseAdunitPlaceTypes[j] > 0)
				{
					for (k = 0; k < g_allParams.Nplace[j]; k++)
						AdUnits[Mcells[Places[j][k].mcell].adunit].NP++;
				}

		}
	}
	fprintf(stderr, "Places intialised.\n");


	//Set up the population for digital contact tracing here... - ggilani 09/03/20
	if (g_allParams.DoDigitalContactTracing)
	{
		g_allParams.NDigitalContactUsers = 0;
		l = m=0;
		//if clustering by Households
		if (g_allParams.DoHouseholds && g_allParams.ClusterDigitalContactUsers)
		{
			//Loop through households

			//NOTE: Are we still okay with this kind of openmp parallelisation. I know there have been some discussions re:openmp, but not followed them completely
			l = m = 0;
#pragma omp parallel for private(tn,i,i1,i2,j,age) schedule(static,1) reduction(+:l,m)
			for (tn = 0; tn < g_allParams.NumThreads; tn++)
			{
				for (i = tn; i < g_allParams.housholdCount; i += g_allParams.NumThreads)
				{
					if (ranf_mt(tn) < g_allParams.PropPopUsingDigitalContactTracing)
					{
						//select this household for digital contact app use
						//loop through household members and check whether they will be selected for use
						i1 = Households[i].FirstPerson;
						i2 = i1 + Households[i].nh;
						for (j = i1; j < i2; j++)
						{
							//get age of host
							age = HOST_AGE_GROUP(j);
							if (age >= NUM_AGE_GROUPS) age = NUM_AGE_GROUPS - 1;
							//check to see if host will be a user based on age group
							if (ranf_mt(tn) < g_allParams.ProportionSmartphoneUsersByAge[age])
							{
								Hosts[j].digitalContactTracingUser = 1;
								l++;
							}
						}
						m++;
					}
				}
			}
			g_allParams.NDigitalContactUsers = l;
			g_allParams.NDigitalHouseholdUsers = m;
			fprintf(stderr, "Number of digital contact tracing households: %i, out of total number of households: %i\n", g_allParams.NDigitalHouseholdUsers, g_allParams.housholdCount);
			fprintf(stderr, "Number of digital contact tracing users: %i, out of population size: %i\n", g_allParams.NDigitalContactUsers, g_allParams.populationSize);
		}
		else // Just go through the population and assign people to the digital contact tracing app based on probability by age.
		{
			//for use with non-clustered
			l = 0;
#pragma omp parallel for private(tn,i,i1,i2,j,age) schedule(static,1) reduction(+:l)
			for (tn = 0; tn < g_allParams.NumThreads; tn++)
			{
				for (i = tn; i < g_allParams.populationSize; i += g_allParams.NumThreads)
				{
					age = HOST_AGE_GROUP(i);
					if (age >= NUM_AGE_GROUPS) age = NUM_AGE_GROUPS - 1;

					if (ranf_mt(tn) < (g_allParams.ProportionSmartphoneUsersByAge[age] * g_allParams.PropPopUsingDigitalContactTracing))
					{
						Hosts[i].digitalContactTracingUser = 1;
						l++;
					}
				}
			}
			g_allParams.NDigitalContactUsers = l;
			fprintf(stderr, "Number of digital contact tracing users: %i, out of population size: %i\n", g_allParams.NDigitalContactUsers, g_allParams.populationSize);
		}
	}


	UpdateProbs(0);
	if (g_allParams.DoAirports) SetupAirports();
	if (g_allParams.R0scale != 1.0)
	{
		g_allParams.HouseholdTrans *= g_allParams.R0scale;
		g_allParams.R0 *= g_allParams.R0scale;
		for (j = 0; j < g_allParams.PlaceTypeNum; j++)
			g_allParams.PlaceTypeTrans[j] *= g_allParams.R0scale;
		fprintf(stderr, "Rescaled transmission coefficients by factor of %lg\n", g_allParams.R0scale);
	}
	t = s = t2 = 0;
	for (i = 0; i < MAX_HOUSEHOLD_SIZE; i++)
	{
		t += ((double)(i + 1)) * (g_allParams.HouseholdSizeDistrib[0][i] - t2);
		t2 = g_allParams.HouseholdSizeDistrib[0][i];
	}
	t2 = s = 0;
	s3 = 1.0;

#pragma omp parallel for private(i,s2,j,k,q,l,d,y,m,tn) schedule(static,1) reduction(+:s,t2) //schedule(static,1000)
	for (tn = 0; tn < g_allParams.NumThreads; tn++) //changed this looping to allow for multi-threaded random numbers
	{
		for (i = tn; i < g_allParams.populationSize; i += g_allParams.NumThreads)
		{
			if (g_allParams.InfectiousnessSD == 0)
				Hosts[i].infectiousness = (float)g_allParams.AgeInfectiousness[HOST_AGE_GROUP(i)];
			else
				Hosts[i].infectiousness = (float)(g_allParams.AgeInfectiousness[HOST_AGE_GROUP(i)] * gen_gamma_mt(g_allParams.InfectiousnessGamA, g_allParams.InfectiousnessGamR, tn)); //made this multi-threaded: 28/11/14
			q = g_allParams.ProportionSymptomatic[HOST_AGE_GROUP(i)];
			if (ranf_mt(tn) < q) //made this multi-threaded: 28/11/14
				Hosts[i].infectiousness = (float)(-g_allParams.SymptInfectiousness * Hosts[i].infectiousness);
			j = (int)floor((q = ranf_mt(tn) * CDF_RES)); //made this multi-threaded: 28/11/14
			q -= ((double)j);
			Hosts[i].recovery_or_death_time = (unsigned short int) floor(0.5 - (g_allParams.InfectiousPeriod * log(q * g_allParams.infectious_icdf[j + 1] + (1.0 - q) * g_allParams.infectious_icdf[j]) / g_allParams.TimeStep));

			if (g_allParams.DoHouseholds)
			{
				s2 = g_allParams.TimeStep * g_allParams.HouseholdTrans * fabs(Hosts[i].infectiousness) * g_allParams.HouseholdDenomLookup[Households[Hosts[i].hh].nhr - 1];
				d = 1.0; l = (int)Hosts[i].recovery_or_death_time;
				for (k = 0; k < l; k++) { y = 1.0 - s2 * g_allParams.infectiousness[k]; d *= ((y < 0) ? 0 : y); }
				l = Households[Hosts[i].hh].FirstPerson;
				m = l + Households[Hosts[i].hh].nh;
				for (k = l; k < m; k++) if ((Hosts[k].inf == InfStat_Susceptible) && (k != i)) s += (1 - d) * g_allParams.AgeSusceptibility[HOST_AGE_GROUP(i)];
			}
			q = (g_allParams.LatentToSymptDelay > Hosts[i].recovery_or_death_time * g_allParams.TimeStep) ? Hosts[i].recovery_or_death_time * g_allParams.TimeStep : g_allParams.LatentToSymptDelay;
			s2 = fabs(Hosts[i].infectiousness) * g_allParams.RelativeSpatialContact[HOST_AGE_GROUP(i)] * g_allParams.TimeStep;
			l = (int)(q / g_allParams.TimeStep);
			for (k = 0; k < l; k++) t2 += s2 * g_allParams.infectiousness[k];
			s2 *= ((Hosts[i].infectiousness < 0) ? g_allParams.SymptSpatialContactRate : 1);
			l = (int)Hosts[i].recovery_or_death_time;
			for (; k < l; k++) t2 += s2 * g_allParams.infectiousness[k];

		}
	}
	t2 *= (s3 / ((double)g_allParams.populationSize));
	s /= ((double)g_allParams.populationSize);
	fprintf(stderr, "Household mean size=%lg\nHousehold R0=%lg\n", t, g_allParams.R0household = s);
	t = x = y = 0;
	if (g_allParams.DoPlaces)
		for (j = 0; j < g_allParams.PlaceTypeNum; j++)
			if (j != g_allParams.HotelPlaceType)
			{
#pragma omp parallel for private(i,k,d,q,s2,s3,t3,l,m,x,y) schedule(static,1000) reduction(+:t)
				for (i = 0; i < g_allParams.populationSize; i++)
				{
					k = Hosts[i].PlaceLinks[j];
					if (k >= 0)
					{
						q = (g_allParams.LatentToSymptDelay > Hosts[i].recovery_or_death_time * g_allParams.TimeStep) ? Hosts[i].recovery_or_death_time * g_allParams.TimeStep : g_allParams.LatentToSymptDelay;
						s2 = fabs(Hosts[i].infectiousness) * g_allParams.TimeStep * g_allParams.PlaceTypeTrans[j];
						x = s2 / g_allParams.PlaceTypeGroupSizeParam1[j];
						d = 1.0; l = (int)(q / g_allParams.TimeStep);
						for (m = 0; m < l; m++) { y = 1.0 - x * g_allParams.infectiousness[m]; d *= ((y < 0) ? 0 : y); }
						s3 = ((double)(Places[j][k].group_size[Hosts[i].PlaceGroupLinks[j]] - 1));
						x *= ((Hosts[i].infectiousness < 0) ? (g_allParams.SymptPlaceTypeContactRate[j] * (1 - g_allParams.SymptPlaceTypeWithdrawalProp[j])) : 1);
						l = (int)Hosts[i].recovery_or_death_time;
						for (; m < l; m++) { y = 1.0 - x * g_allParams.infectiousness[m]; d *= ((y < 0) ? 0 : y); }

						t3 = d;
						x = g_allParams.PlaceTypePropBetweenGroupLinks[j] * s2 / ((double)Places[j][k].n);
						d = 1.0; l = (int)(q / g_allParams.TimeStep);
						for (m = 0; m < l; m++) { y = 1.0 - x * g_allParams.infectiousness[m]; d *= ((y < 0) ? 0 : y); }
						x *= ((Hosts[i].infectiousness < 0) ? (g_allParams.SymptPlaceTypeContactRate[j] * (1 - g_allParams.SymptPlaceTypeWithdrawalProp[j])) : 1);
						l = (int)Hosts[i].recovery_or_death_time;
						for (; m < l; m++) { y = 1.0 - x * g_allParams.infectiousness[m]; d *= ((y < 0) ? 0 : y); }
						t += (1 - t3 * d) * s3 + (1 - d) * (((double)(Places[j][k].n - 1)) - s3);
					}
				}
				fprintf(stderr, "%lg  ", t / ((double)g_allParams.populationSize));
			}
	{
		double recovery_time_days = 0;
		double recovery_time_timesteps = 0;
#pragma omp parallel for private(i) schedule(static,500) reduction(+:recovery_time_days,recovery_time_timesteps)
		for (i = 0; i < g_allParams.populationSize; i++)
		{
			recovery_time_days += Hosts[i].recovery_or_death_time * g_allParams.TimeStep;
			recovery_time_timesteps += Hosts[i].recovery_or_death_time;
			Hosts[i].recovery_or_death_time = 0;
		}
		t /= ((double)g_allParams.populationSize);
		recovery_time_days /= ((double)g_allParams.populationSize);
		recovery_time_timesteps /= ((double)g_allParams.populationSize);
		fprintf(stderr, "R0 for places = %lg\nR0 for random spatial = %lg\nOverall R0=%lg\n", g_allParams.R0places = t, g_allParams.R0spatial = g_allParams.R0 - s - t, g_allParams.R0);
		fprintf(stderr, "Mean infectious period (sampled) = %lg (%lg)\n", recovery_time_days, recovery_time_timesteps);
	}
	if (g_allParams.DoSI)
		g_allParams.LocalBeta = (g_allParams.R0 / t2 - s - t);
	else
		g_allParams.LocalBeta = (g_allParams.R0 - s - t) / t2;
	if ((g_allParams.LocalBeta < 0) || (!g_allParams.DoSpatial))
	{
		g_allParams.LocalBeta = g_allParams.R0spatial = 0;
		fprintf(stderr, "Reset spatial R0 to 0\n");
	}
	fprintf(stderr, "LocalBeta = %lg\n", g_allParams.LocalBeta);
	TSMean = TSMeanNE; TSVar = TSVarNE;
	fprintf(stderr, "Calculated approx cell probabilities\n");
	for (i = 0; i < INFECT_TYPE_MASK; i++) inftype_av[i] = 0;
	for (i = 0; i < MAX_COUNTRIES; i++) infcountry_av[i] = infcountry_num[i] = 0;
	for (i = 0; i < MAX_SEC_REC; i++)
		for (j = 0; j < MAX_GEN_REC; j++)
			indivR0_av[i][j] = 0;
	for (i = 0; i <= MAX_HOUSEHOLD_SIZE; i++)
		for (j = 0; j <= MAX_HOUSEHOLD_SIZE; j++)
			inf_household_av[i][j] = case_household_av[i][j] = 0;
	DoInitUpdateProbs = 1;
	for (i = 0; i < g_allParams.cellCount; i++)	Cells[i].tot_treat = 1;  //This makes sure InitModel intialises the cells.
	g_allParams.NRactE = g_allParams.NRactNE = 0;
	for (i = 0; i < g_allParams.populationSize; i++) Hosts[i].esocdist_comply = (ranf() < g_allParams.EnhancedSocDistProportionCompliant[HOST_AGE_GROUP(i)]) ? 1 : 0;
	if (!g_allParams.EnhancedSocDistClusterByHousehold)
	{
		for (i = 0; i < g_allParams.housholdCount;i++)
		{
			l = Households[i].FirstPerson;
			m = l + Households[i].nh;
			i2 = 0;
			for (k = l; k < m; k++) if (Hosts[k].esocdist_comply) i2=1;
			if (i2)
				for (k = l; k < m; k++) Hosts[k].esocdist_comply = 1;
		}
	}

	if (g_allParams.OutputBitmap)
	{
		InitBMHead();
	}
	if (g_allParams.DoMassVacc)
	{
		if (!(State.mvacc_queue = (int*)calloc(g_allParams.populationSize, sizeof(int)))) ERR_CRITICAL("Unable to allocate host storage\n");
		for (i = j = 0; i < g_allParams.populationSize; i++)
		{
			if ((HOST_AGE_YEAR(i) >= g_allParams.VaccPriorityGroupAge[0]) && (HOST_AGE_YEAR(i) <= g_allParams.VaccPriorityGroupAge[1]))
			{
				if (ranf() < g_allParams.VaccProp)
					State.mvacc_queue[j++] = i;
			}
		}
		k = j;
		for (i = 0; i < g_allParams.populationSize; i++)
		{
			if ((HOST_AGE_YEAR(i) < g_allParams.VaccPriorityGroupAge[0]) || (HOST_AGE_YEAR(i) > g_allParams.VaccPriorityGroupAge[1]))
			{
				if (ranf() < g_allParams.VaccProp)
					State.mvacc_queue[j++] = i;
			}
		}
		State.n_mvacc = j;
		fprintf(stderr, "Number to be vaccinated=%i\n", State.n_mvacc);
		for (i = 0; i < 2; i++)
		{
			for (j = 0; j < k; j++)
			{
				l = (int)(ranf() * ((double)k));
				m = State.mvacc_queue[j];
				State.mvacc_queue[j] = State.mvacc_queue[l];
				State.mvacc_queue[l] = m;
			}
			for (j = k; j < State.n_mvacc; j++)
			{
				l = k + ((int)(ranf() * ((double)(State.n_mvacc - k))));
				m = State.mvacc_queue[j];
				State.mvacc_queue[j] = State.mvacc_queue[l];
				State.mvacc_queue[l] = m;
			}
		}
		fprintf(stderr, "Configured mass vaccination queue.\n");
	}
	PeakHeightSum = PeakHeightSS = PeakTimeSum = PeakTimeSS = 0;
	i = (g_allParams.ncw / 2) * g_allParams.nch + g_allParams.nch / 2;
	j = (g_allParams.ncw / 2 + 2) * g_allParams.nch + g_allParams.nch / 2;
	fprintf(stderr, "UTM dist horiz=%lg %lg\n", sqrt(dist2_cc(Cells + i, Cells + j)), sqrt(dist2_cc(Cells + j, Cells + i)));
	j = (g_allParams.ncw / 2) * g_allParams.nch + g_allParams.nch / 2 + 2;
	fprintf(stderr, "UTM dist vert=%lg %lg\n", sqrt(dist2_cc(Cells + i, Cells + j)), sqrt(dist2_cc(Cells + j, Cells + i)));
	j = (g_allParams.ncw / 2 + 2) * g_allParams.nch + g_allParams.nch / 2 + 2;
	fprintf(stderr, "UTM dist diag=%lg %lg\n", sqrt(dist2_cc(Cells + i, Cells + j)), sqrt(dist2_cc(Cells + j, Cells + i)));

	//if(P.OutputBitmap)
	//{
	//	CaptureBitmap();
	//	OutputBitmap(0);
	//}
	fprintf(stderr, "Model configuration complete.\n");
}

void SetupPopulation(char* DensityFile, char* SchoolFile, char* RegDemogFile)
{
	int i, j, k, l, m, i2, j2, last_i, mr, ad, tn, *mcl, country;
	unsigned int rn, rn2;
	double t, s, x, y, xh, yh, maxd, CumAgeDist[NUM_AGE_GROUPS + 1];
	char buf[4096], *col;
	const char delimiters[] = " \t,";
	FILE* dat = NULL, *dat2;
	bin_file rec;
	double *mcell_dens;
	int *mcell_adunits, *mcell_num, *mcell_country;

	if (!(Cells = (cell*)calloc(g_allParams.cellCount, sizeof(cell)))) ERR_CRITICAL("Unable to allocate cell storage\n");
	if (!(Mcells = (microcell*)calloc(g_allParams.microcellCount, sizeof(microcell)))) ERR_CRITICAL("Unable to allocate cell storage\n");
	if (!(mcell_num = (int*)malloc(g_allParams.microcellCount * sizeof(int)))) ERR_CRITICAL("Unable to allocate cell storage\n");
	if (!(mcell_dens = (double*)malloc(g_allParams.microcellCount * sizeof(double)))) ERR_CRITICAL("Unable to allocate cell storage\n");
	if (!(mcell_country = (int*)malloc(g_allParams.microcellCount * sizeof(int)))) ERR_CRITICAL("Unable to allocate cell storage\n");
	if (!(mcell_adunits = (int*)malloc(g_allParams.microcellCount * sizeof(int)))) ERR_CRITICAL("Unable to allocate cell storage\n");

	for (j = 0; j < g_allParams.microcellCount; j++)
	{
		Mcells[j].n = 0;
		mcell_adunits[j] = -1;
		mcell_dens[j] = 0;
		mcell_num[j] = mcell_country[j] = 0;
	}
	if (g_allParams.DoAdUnits)
		for (i = 0; i < MAX_ADUNITS; i++)
			g_allParams.PopByAdunit[i][0] = g_allParams.PopByAdunit[i][1] = 0;
	if (g_allParams.DoHeteroDensity)
	{
		if (!g_allParams.DoAdunitBoundaries) g_allParams.NumAdunits = 0;
		//		if(!(dat2=fopen("EnvTest.txt","w"))) ERR_CRITICAL("Unable to open test file\n");
		fprintf(stderr, "Density file contains %i datapoints.\n", (int)g_allParams.binFileLen);
		for (rn = rn2 = mr = 0; rn < g_allParams.binFileLen; rn++)
		{
			if (g_allParams.DoAdUnits)
			{
				x = BF[rn].x; y = BF[rn].y; t = BF[rn].pop; country = BF[rn].cnt; j2 = BF[rn].ad; //changed from i to rn to loop over indices properly
				rec = BF[rn];
				m = (j2 % g_allParams.AdunitLevel1Mask) / g_allParams.AdunitLevel1Divisor;
				if (g_allParams.DoAdunitBoundaries)
				{
					if (g_allParams.AdunitLevel1Lookup[m] >= 0)
					{
						if (j2 / g_allParams.AdunitLevel1Mask == AdUnits[g_allParams.AdunitLevel1Lookup[m]].id / g_allParams.AdunitLevel1Mask)
						{
							k = 1;
							AdUnits[g_allParams.AdunitLevel1Lookup[m]].cnt_id = country;
						}
						else
							k = 0;
					}
					else
						k = 0;
				}
				else
				{
					k = 1;
					if (g_allParams.AdunitLevel1Lookup[m] < 0)
					{
						g_allParams.AdunitLevel1Lookup[m] = g_allParams.NumAdunits;
						AdUnits[g_allParams.NumAdunits].id = j2;
						AdUnits[g_allParams.NumAdunits].cnt_id = country;
						g_allParams.NumAdunits++;
						if (g_allParams.NumAdunits >= MAX_ADUNITS) ERR_CRITICAL("Total number of administrative units exceeds MAX_ADUNITS\n");
					}
					else
					{
						AdUnits[g_allParams.AdunitLevel1Lookup[m]].cnt_id = country;
					}
				}
			}
			else
			{
				k = 1;
				x = BF[i].x; y = BF[i].y; t = BF[i].pop; country = BF[i].cnt; j2 = BF[i].ad;
				rec = BF[rn];
			}
			if ((k) && (x >= g_allParams.SpatialBoundingBox[0]) && (y >= g_allParams.SpatialBoundingBox[1]) && (x < g_allParams.SpatialBoundingBox[2]) && (y < g_allParams.SpatialBoundingBox[3]))
			{
				j = (int)floor((x - g_allParams.SpatialBoundingBox[0]) / g_allParams.mcwidth + 0.1);
				k = (int)floor((y - g_allParams.SpatialBoundingBox[1]) / g_allParams.mcheight + 0.1);
				l = j * g_allParams.nmch + k;
				if (l < g_allParams.microcellCount)
				{
					mr++;
					mcell_dens[l] += t;
					mcell_country[l] = country;
					//fprintf(stderr,"mcell %i, country %i, pop %lg\n",l,country,t);
					mcell_num[l]++;
					if (g_allParams.DoAdUnits)
					{
						mcell_adunits[l] = g_allParams.AdunitLevel1Lookup[m];
						if (mcell_adunits[l] < 0) fprintf(stderr, "Cell %i has adunits<0\n", l);
						g_allParams.PopByAdunit[g_allParams.AdunitLevel1Lookup[m]][0] += t;
					}
					else
						mcell_adunits[l] = 0;
					if ((g_allParams.OutputDensFile) && (g_allParams.DoBin) && (mcell_adunits[l] >= 0))
					{
						if (rn2 < rn) BF[rn2] = rec;
						rn2++;
					}
				}
			}
		}
		//		fclose(dat2);
		fprintf(stderr, "%i valid mcells read from density file.\n", mr);
		if ((g_allParams.OutputDensFile) && (g_allParams.DoBin)) g_allParams.binFileLen = rn2;
		if (g_allParams.DoBin == 0)
		{
			if (g_allParams.OutputDensFile)
			{
				free(BinFileBuf);
				g_allParams.DoBin = 1;
				g_allParams.binFileLen = 0;
				for (l = 0; l < g_allParams.microcellCount; l++)
					if (mcell_adunits[l] >= 0) g_allParams.binFileLen++;
				if (!(BinFileBuf = (void*)malloc(g_allParams.binFileLen * sizeof(bin_file)))) ERR_CRITICAL("Unable to allocate binary file buffer\n");
				BF = (bin_file*)BinFileBuf;
				fprintf(stderr, "Binary density file should contain %i cells.\n", (int)g_allParams.binFileLen);
				rn = 0;
				for (l = 0; l < g_allParams.microcellCount; l++)
					if (mcell_adunits[l] >= 0)
					{
						BF[rn].x = (double)(g_allParams.mcwidth * (((double)(l / g_allParams.nmch)) + 0.5)) + g_allParams.SpatialBoundingBox[0]; //x
						BF[rn].y = (double)(g_allParams.mcheight * (((double)(l % g_allParams.nmch)) + 0.5)) + g_allParams.SpatialBoundingBox[1]; //y
						BF[rn].ad = (g_allParams.DoAdUnits) ? (AdUnits[mcell_adunits[l]].id) : 0;
						BF[rn].pop = mcell_dens[l];
						BF[rn].cnt = mcell_country[l];
						rn++;
					}
			}
		}

		if (g_allParams.OutputDensFile)
		{
			if (!(dat2 = fopen(OutDensFile, "wb"))) ERR_CRITICAL("Unable to open output density file\n");
			rn = 0xf0f0f0f0;
			fwrite_big((void*)& rn, sizeof(unsigned int), 1, dat2);
			fprintf(stderr, "Saving population density file with NC=%i...\n", (int)g_allParams.binFileLen);
			fwrite_big((void*) & (g_allParams.binFileLen), sizeof(unsigned int), 1, dat2);
			fwrite_big(BinFileBuf, sizeof(bin_file), (size_t)g_allParams.binFileLen, dat2);
			fclose(dat2);
		}
		free(BinFileBuf);
		fprintf(stderr, "Population files read.\n");
		maxd = 0;
		for (i = 0; i < g_allParams.microcellCount; i++)
		{
			if (mcell_num[i] > 0)
			{
				mcell_dens[i] /= ((double)mcell_num[i]);
				Mcells[i].country = (unsigned short)mcell_country[i];
				if (g_allParams.DoAdUnits)
					Mcells[i].adunit = mcell_adunits[i];
				else
					Mcells[i].adunit = 0;
			}
			else
				Mcells[i].adunit = -1;
			maxd += mcell_dens[i];
		}
	}
	else
	{
		for (i = 0; i < g_allParams.microcellCount; i++)
		{
			mcell_dens[i] = 1.0;
			Mcells[i].country = 1;
		}
		maxd = ((double)g_allParams.microcellCount);
	}
	if (!g_allParams.DoAdUnits) g_allParams.NumAdunits = 1;
	if ((g_allParams.DoAdUnits) && (g_allParams.DoAdunitDemog))
	{
		if (!(State.InvAgeDist = (int**)malloc(g_allParams.NumAdunits * sizeof(int*)))) ERR_CRITICAL("Unable to allocate InvAgeDist storage\n");
		for (i = 0; i < g_allParams.NumAdunits; i++)
			if (!(State.InvAgeDist[i] = (int*)malloc(1000 * sizeof(int)))) ERR_CRITICAL("Unable to allocate InvAgeDist storage\n");
		if (!(dat = fopen(RegDemogFile, "rb"))) ERR_CRITICAL("Unable to open regional demography file\n");
		for (k = 0; k < g_allParams.NumAdunits; k++)
		{
			for (i = 0; i < NUM_AGE_GROUPS; i++)
				g_allParams.PropAgeGroup[k][i] = 0;
			for (i = 0; i < MAX_HOUSEHOLD_SIZE; i++)
				g_allParams.HouseholdSizeDistrib[k][i] = 0;
			g_allParams.PopByAdunit[k][1] = 0;
		}
		while (!feof(dat))
		{
			fgets(buf, 2047, dat);
			col = strtok(buf, delimiters);
			sscanf(col, "%i", &l);
			m = (l % g_allParams.AdunitLevel1Mask) / g_allParams.AdunitLevel1Divisor;
			k = g_allParams.AdunitLevel1Lookup[m];
			if (k >= 0)
				if (l / g_allParams.AdunitLevel1Mask == AdUnits[k].id / g_allParams.AdunitLevel1Mask)
				{
					col = strtok(NULL, delimiters);
					sscanf(col, "%lg", &x);
					g_allParams.PopByAdunit[k][1] += x;
					t = 0;
					for (i = 0; i < NUM_AGE_GROUPS; i++)
					{
						col = strtok(NULL, delimiters);
						sscanf(col, "%lg", &s);
						g_allParams.PropAgeGroup[k][i] += s;
					}
					col = strtok(NULL, delimiters);
					if (g_allParams.DoHouseholds)
					{
						sscanf(col, "%lg", &y);
						for (i = 0; i < MAX_HOUSEHOLD_SIZE; i++)
						{
							col = strtok(NULL, delimiters);
							sscanf(col, "%lg", &s);
							g_allParams.HouseholdSizeDistrib[k][i] += y * s;
						}
					}
				}
		}
		fclose(dat);
		for (k = 0; k < g_allParams.NumAdunits; k++)
		{
			t = 0;
			for (i = 0; i < NUM_AGE_GROUPS; i++)
				t += g_allParams.PropAgeGroup[k][i];
			CumAgeDist[0] = 0;
			for (i = 1; i <= NUM_AGE_GROUPS; i++)
			{
				g_allParams.PropAgeGroup[k][i - 1] /= t;
				CumAgeDist[i] = CumAgeDist[i - 1] + g_allParams.PropAgeGroup[k][i - 1];
			}
			for (i = j = 0; i < 1000; i++)
			{
				t = ((double)i) / 1000;
				while (t >= CumAgeDist[j + 1]) j++;
				t = AGE_GROUP_WIDTH * (((double)j) + (t - CumAgeDist[j]) / (CumAgeDist[j + 1] - CumAgeDist[j]));
				State.InvAgeDist[k][i] = (int)t;
			}
			State.InvAgeDist[k][1000 - 1] = NUM_AGE_GROUPS * AGE_GROUP_WIDTH - 1;
			if (g_allParams.DoHouseholds)
			{
				t = 0;
				for (i = 0; i < MAX_HOUSEHOLD_SIZE; i++)
					t += g_allParams.HouseholdSizeDistrib[k][i];
				g_allParams.HouseholdSizeDistrib[k][0] /= t;
				for (i = 1; i < MAX_HOUSEHOLD_SIZE - 1; i++)
					g_allParams.HouseholdSizeDistrib[k][i] = g_allParams.HouseholdSizeDistrib[k][i] / t + g_allParams.HouseholdSizeDistrib[k][i - 1];
				g_allParams.HouseholdSizeDistrib[k][MAX_HOUSEHOLD_SIZE - 1] = 1.0;
			}
			else
			{
				for (i = 0; i < MAX_HOUSEHOLD_SIZE - 1; i++)
					g_allParams.HouseholdSizeDistrib[k][i] = 1.0;
			}
		}
	}
	else
	{
		if (!(State.InvAgeDist = (int**)malloc(sizeof(int*)))) ERR_CRITICAL("Unable to allocate InvAgeDist storage\n");
		if (!(State.InvAgeDist[0] = (int*)malloc(1000 * sizeof(int)))) ERR_CRITICAL("Unable to allocate InvAgeDist storage\n");
		CumAgeDist[0] = 0;
		for (i = 1; i <= NUM_AGE_GROUPS; i++)
			CumAgeDist[i] = CumAgeDist[i - 1] + g_allParams.PropAgeGroup[0][i - 1];
		for (i = j = 0; i < 1000; i++)
		{
			t = ((double)i) / 1000;
			if (t >= CumAgeDist[j + 1]) j++;
			t = AGE_GROUP_WIDTH * (((double)j) + (t - CumAgeDist[j]) / (CumAgeDist[j + 1] - CumAgeDist[j]));
			State.InvAgeDist[0][i] = (int)t;
		}
		State.InvAgeDist[0][1000 - 1] = NUM_AGE_GROUPS * AGE_GROUP_WIDTH - 1;
	}
	if (g_allParams.DoAdUnits)
		for (i = 0; i < g_allParams.NumAdunits; i++) AdUnits[i].n = 0;
	if ((g_allParams.DoAdUnits) && (g_allParams.DoAdunitDemog) && (g_allParams.DoCorrectAdunitPop))
	{
		for (i = 0; i < g_allParams.NumAdunits; i++)
			fprintf(stderr, "%i\t%i\t%lg\t%lg\n", i, (AdUnits[i].id % g_allParams.AdunitLevel1Mask) / g_allParams.AdunitLevel1Divisor, g_allParams.PropAgeGroup[i][0], g_allParams.HouseholdSizeDistrib[i][0]);
		maxd = 0;
		for (i = 0; i < g_allParams.microcellCount; i++)
		{
			if (mcell_num[i] > 0)
      {
				if (mcell_adunits[i] < 0) ERR_CRITICAL_FMT("Cell %i has adunits < 0 (indexing PopByAdunit)\n", i);
				mcell_dens[i] *= g_allParams.PopByAdunit[mcell_adunits[i]][1] / (1e-10 + g_allParams.PopByAdunit[mcell_adunits[i]][0]);
      }
			maxd += mcell_dens[i];
		}
		t = 0;
		for (i = 0; i < g_allParams.NumAdunits; i++)
			t += g_allParams.PopByAdunit[i][1];
		i = g_allParams.populationSize;
		g_allParams.populationSize = (int)t;
		fprintf(stderr, "Population size reset from %i to %i\n", i, g_allParams.populationSize);
	}
	t = 1.0;
	for (i =m= 0; i < (g_allParams.microcellCount - 1); i++)
	{
		s = mcell_dens[i] / maxd / t;
		if (s > 1.0) s = 1.0;
		m += (Mcells[i].n = (int)ignbin_mt((long)(g_allParams.populationSize - m), s, 0));
		t -= mcell_dens[i] / maxd;
		if (Mcells[i].n > 0) {
			g_allParams.NMCP++;
			if (mcell_adunits[i] < 0) ERR_CRITICAL_FMT("Cell %i has adunits < 0 (indexing AdUnits)\n", i);
			AdUnits[mcell_adunits[i]].n += Mcells[i].n;
		}
	}
	Mcells[g_allParams.microcellCount - 1].n = g_allParams.populationSize - m;
	if (Mcells[g_allParams.microcellCount - 1].n > 0)
	{
		g_allParams.NMCP++;
		AdUnits[mcell_adunits[g_allParams.microcellCount - 1]].n += Mcells[g_allParams.microcellCount - 1].n;
	}

	free(mcell_dens);
	free(mcell_num);
	free(mcell_country);
	free(mcell_adunits);
	t = 0.0;

	if (!(McellLookup = (microcell * *)malloc(g_allParams.NMCP * sizeof(microcell*)))) ERR_CRITICAL("Unable to allocate cell storage\n");
	if (!(mcl = (int*)malloc(g_allParams.populationSize * sizeof(int)))) ERR_CRITICAL("Unable to allocate cell storage\n");
	State.CellMemberArray = mcl;
	g_allParams.populatedCellCount = 0;
	for (i = i2 = j2 = 0; i < g_allParams.cellCount; i++)
	{
		Cells[i].n = 0;
		k = (i / g_allParams.nch) * g_allParams.microcellsOnACellSide * g_allParams.nmch + (i % g_allParams.nch) * g_allParams.microcellsOnACellSide;
		Cells[i].members = mcl + j2;
		for (l = 0; l < g_allParams.microcellsOnACellSide; l++)
			for (m = 0; m < g_allParams.microcellsOnACellSide; m++)
			{
				j = k + m + l * g_allParams.nmch;
				if (Mcells[j].n > 0)
				{
					Mcells[j].members = mcl + j2;
					//if(!(Mcells[j].members=(int *) calloc(Mcells[j].n,sizeof(int)))) ERR_CRITICAL("Unable to allocate cell storage\n"); //replaced line above with this to ensure members don't get mixed across microcells
					McellLookup[i2++] = Mcells + j;
					Cells[i].n += Mcells[j].n;
					j2 += Mcells[j].n;
				}
			}
		if (Cells[i].n > 0) g_allParams.populatedCellCount++;
	}
	fprintf(stderr, "Number of hosts assigned = %i\n", j2);
	if (!g_allParams.DoAdUnits) g_allParams.AdunitLevel1Lookup[0] = 0;
	fprintf(stderr, "Number of cells with non-zero population = %i\n", g_allParams.populatedCellCount);
	fprintf(stderr, "Number of microcells with non-zero population = %i\n", g_allParams.NMCP);

	if (!(CellLookup = (cell * *)malloc(g_allParams.populatedCellCount * sizeof(cell*)))) ERR_CRITICAL("Unable to allocate cell storage\n");
	if (!(mcl = (int*)malloc(g_allParams.populationSize * sizeof(int)))) ERR_CRITICAL("Unable to allocate cell storage\n");
	State.CellSuscMemberArray = mcl;
	i2 = k = 0;
	for (j = 0; j < g_allParams.cellCount; j++)
		if (Cells[j].n > 0)
		{
			CellLookup[i2++] = Cells + j;
			Cells[j].susceptible = mcl + k;
			k += Cells[j].n;
		}
	if (i2 > g_allParams.populatedCellCount) fprintf(stderr, "######## Over-run on CellLookup array NCP=%i i2=%i ###########\n", g_allParams.populatedCellCount, i2);
	i2 = 0;

	if (!(Hosts = (person*)calloc(g_allParams.populationSize, sizeof(person)))) ERR_CRITICAL("Unable to allocate host storage\n");
	fprintf(stderr, "sizeof(person)=%i\n", (int) sizeof(person));
	for (i = 0; i < g_allParams.populatedCellCount; i++)
	{
		cell *c = CellLookup[i];
		if (c->n > 0)
		{
			if (!(c->InvCDF = (int*)malloc(1025 * sizeof(int)))) ERR_CRITICAL("Unable to allocate cell storage\n");
			if (!(c->max_trans = (float*)malloc(g_allParams.populatedCellCount * sizeof(float)))) ERR_CRITICAL("Unable to allocate cell storage\n");
			if (!(c->cum_trans = (float*)malloc(g_allParams.populatedCellCount * sizeof(float)))) ERR_CRITICAL("Unable to allocate cell storage\n");
		}
	}
	for (i = 0; i < g_allParams.cellCount; i++)
	{
		Cells[i].cumTC = 0;
		for (j = 0; j < Cells[i].n; j++) Cells[i].members[j] = -1;
	}
	fprintf(stderr, "Cells assigned\n");
	for (i = 0; i <= MAX_HOUSEHOLD_SIZE; i++) denom_household[i] = 0;
	g_allParams.housholdCount = 0;
	for (i = j2 = 0; j2 < g_allParams.NMCP; j2++)
	{
		j = (int)(McellLookup[j2] - Mcells);
		l = ((j / g_allParams.nmch) / g_allParams.microcellsOnACellSide) * g_allParams.nch + ((j % g_allParams.nmch) / g_allParams.microcellsOnACellSide);
		ad = ((g_allParams.DoAdunitDemog) && (g_allParams.DoAdUnits)) ? Mcells[j].adunit : 0;
		for (k = 0; k < Mcells[j].n;)
		{
			m = 1;
			if (g_allParams.DoHouseholds)
			{
				s = ranf_mt(0);
				while ((s > g_allParams.HouseholdSizeDistrib[ad][m - 1]) && (k + m < Mcells[j].n) && (m < MAX_HOUSEHOLD_SIZE)) m++;
			}
			denom_household[m]++;
			for (i2 = 0; i2 < m; i2++)
			{
				//				fprintf(stderr,"%i ",i+i2);
				Hosts[i + i2].listpos = m; //used temporarily to store household size
				Mcells[j].members[k + i2] = i + i2;
				Cells[l].susceptible[Cells[l].cumTC] = i + i2;
				Cells[l].members[Cells[l].cumTC++] = i + i2;
				Hosts[i + i2].pcell = l;
				Hosts[i + i2].mcell = j;
				Hosts[i + i2].hh = g_allParams.housholdCount;
			}
			g_allParams.housholdCount++;
			i += m;
			k += m;
		}
	}
	if (!(Households = (household*)malloc(g_allParams.housholdCount * sizeof(household)))) ERR_CRITICAL("Unable to allocate household storage\n");
	for (j = 0; j < NUM_AGE_GROUPS; j++) AgeDist[j] = AgeDist2[j] = 0;
	if (g_allParams.DoHouseholds) fprintf(stderr, "Household sizes assigned to %i people\n", i);
#pragma omp parallel for private(tn,j2,j,i,k,x,y,xh,yh,i2,m) schedule(static,1)
	for (tn = 0; tn < g_allParams.NumThreads; tn++)
		for (j2 = tn; j2 < g_allParams.NMCP; j2 += g_allParams.NumThreads)
		{
			j = (int)(McellLookup[j2] - Mcells);
			x = (double)(j / g_allParams.nmch);
			y = (double)(j % g_allParams.nmch);
			i = Mcells[j].members[0];
			if (j % 100 == 0)
				fprintf(stderr, "%i=%i (%i %i)            \r", j, Mcells[j].n, Mcells[j].adunit, (AdUnits[Mcells[j].adunit].id % g_allParams.AdunitLevel1Mask) / g_allParams.AdunitLevel1Divisor);
			for (k = 0; k < Mcells[j].n;)
			{
				m = Hosts[i].listpos;
				xh = g_allParams.mcwidth * (ranf_mt(tn) + x);
				yh = g_allParams.mcheight * (ranf_mt(tn) + y);
				AssignHouseholdAges(m, i, tn);
				for (i2 = 0; i2 < m; i2++) Hosts[i + i2].listpos = 0;
				if (g_allParams.DoHouseholds)
				{
					for (i2 = 0; i2 < m; i2++) {
						Hosts[i + i2].inf = InfStat_Susceptible; //added this so that infection status is set to zero and household r0 is correctly calculated
					}
				}
				Households[Hosts[i].hh].FirstPerson = i;
				Households[Hosts[i].hh].nh = m;
				Households[Hosts[i].hh].nhr = m;
				Households[Hosts[i].hh].loc_x = (float)xh;
				Households[Hosts[i].hh].loc_y = (float)yh;
				i += m;
				k += m;
			}
		}
	if (g_allParams.DoCorrectAgeDist)
	{
		double** AgeDistAd, ** AgeDistCorrF, ** AgeDistCorrB;
		if (!(AgeDistAd = (double**)malloc(MAX_ADUNITS * sizeof(double*)))) ERR_CRITICAL("Unable to allocate temp storage\n");
		if (!(AgeDistCorrF = (double**)malloc(MAX_ADUNITS * sizeof(double*)))) ERR_CRITICAL("Unable to allocate temp storage\n");
		if (!(AgeDistCorrB = (double**)malloc(MAX_ADUNITS * sizeof(double*)))) ERR_CRITICAL("Unable to allocate temp storage\n");
		for (i = 0; i < g_allParams.NumAdunits; i++)
		{
			if (!(AgeDistAd[i] = (double*)malloc((NUM_AGE_GROUPS + 1) * sizeof(double)))) ERR_CRITICAL("Unable to allocate temp storage\n");
			if (!(AgeDistCorrF[i] = (double*)malloc((NUM_AGE_GROUPS + 1) * sizeof(double)))) ERR_CRITICAL("Unable to allocate temp storage\n");
			if (!(AgeDistCorrB[i] = (double*)malloc((NUM_AGE_GROUPS + 1) * sizeof(double)))) ERR_CRITICAL("Unable to allocate temp storage\n");
		}

		// compute AgeDistAd[i][j] = total number of people in adunit i, age group j
		for (i = 0; i < g_allParams.NumAdunits; i++)
			for (j = 0; j < NUM_AGE_GROUPS; j++)
				AgeDistAd[i][j] = 0;
		for (i = 0; i < g_allParams.populationSize; i++)
		{
			k = (g_allParams.DoAdunitDemog) ? Mcells[Hosts[i].mcell].adunit : 0;
			AgeDistAd[k][HOST_AGE_GROUP(i)]++;
		}
		// normalize AgeDistAd[i][j], so it's the proportion of people in adunit i that are in age group j
		k = (g_allParams.DoAdunitDemog) ? g_allParams.NumAdunits : 1;
		for (i = 0; i < k; i++)
		{
			s = 0.0;
			for (j = 0; j < NUM_AGE_GROUPS; j++)
				s += AgeDistAd[i][j];
			for (j = 0; j < NUM_AGE_GROUPS; j++)
				AgeDistAd[i][j] /= s;
		}
		// determine adjustments to be made to match age data in parameters
		for (i = 0; i < k; i++)
		{
			s = t = 0;
			AgeDistCorrB[i][0] = 0;
			for (j = 0; j < NUM_AGE_GROUPS; j++)
			{
				// compute s = the proportion of people that need removing from adunit i, age group j to match age data in parameters
				s = t + AgeDistAd[i][j] - g_allParams.PropAgeGroup[i][j] - AgeDistCorrB[i][j];
				if (s > 0)
				{
					t = AgeDistCorrF[i][j] = s; // people to push up into next age group
					AgeDistCorrB[i][j + 1] = 0;
				}
				else
				{
					t = AgeDistCorrF[i][j] = 0;
					AgeDistCorrB[i][j + 1] = fabs(s); // people to pull down from next age group
				}
				AgeDistCorrF[i][j] /= AgeDistAd[i][j]; // convert from proportion of people in the adunit to proportion of people in the adunit and age group
				AgeDistCorrB[i][j] /= AgeDistAd[i][j];
			}
			// output problematic adjustments (these should be 0.0f)
			//fprintf(stderr, "AgeDistCorrB[%i][0] = %f\n", i, AgeDistCorrB[i][0]); // push down from youngest age group
			//fprintf(stderr, "AgeDistCorrF[%i][NUM_AGE_GROUPS - 1] = %f\n", i, AgeDistCorrF[i][NUM_AGE_GROUPS - 1]); // push up from oldest age group
			//fprintf(stderr, "AgeDistCorrB[%i][NUM_AGE_GROUPS] = %f\n", i, AgeDistCorrB[i][NUM_AGE_GROUPS]); // push down from oldest age group + 1
		}

		// make age adjustments to population
#pragma omp parallel for private(tn,j,i,k,m,s) schedule(static,1)
		for (tn = 0; tn < g_allParams.NumThreads; tn++)
			for (i = tn; i < g_allParams.populationSize; i += g_allParams.NumThreads)
			{
				m = (g_allParams.DoAdunitDemog) ? Mcells[Hosts[i].mcell].adunit : 0;
				j = HOST_AGE_GROUP(i);
				s = ranf_mt(tn);
				// probabilistic age adjustment by one age category (5 years)
				if (s < AgeDistCorrF[m][j])
					Hosts[i].age += 5;
				else if (s < AgeDistCorrF[m][j] + AgeDistCorrB[m][j])
					Hosts[i].age -= 5;
			}
		for (i = 0; i < g_allParams.NumAdunits; i++)
		{
			free(AgeDistAd[i]);
			free(AgeDistCorrF[i]);
			free(AgeDistCorrB[i]);
		}
		free(AgeDistAd);
		free(AgeDistCorrF);
		free(AgeDistCorrB);
	}
	for (i = 0; i < g_allParams.populationSize; i++)
	{
		if (Hosts[i].age >= NUM_AGE_GROUPS * AGE_GROUP_WIDTH)
		{
			ERR_CRITICAL_FMT("Person %i has unexpected age %i\n", i, Hosts[i].age);
		}
		AgeDist[HOST_AGE_GROUP(i)]++;
	}
	fprintf(stderr, "Ages/households assigned\n");

	if (!g_allParams.DoRandomInitialInfectionLoc)
	{
		k = (int)(g_allParams.LocationInitialInfection[0][0] / g_allParams.mcwidth);
		l = (int)(g_allParams.LocationInitialInfection[0][1] / g_allParams.mcheight);
		j = k * g_allParams.nmch + l;

		double rand_r = 0.0; //added these variables so that if initial infection location is empty we can search the 10km neighbourhood to find a suitable cell
		double rand_theta = 0.0;
		int counter = 0;
		if (Mcells[j].n < g_allParams.NumInitialInfections[0])
		{
			while (Mcells[j].n < g_allParams.NumInitialInfections[0] && counter < 100)
			{
				rand_r = ranf(); rand_theta = ranf();
				rand_r = 0.083 * sqrt(rand_r); rand_theta = 2 * PI * rand_theta; //rand_r is multiplied by 0.083 as this is roughly equal to 10km in decimal degrees
				k = (int)((g_allParams.LocationInitialInfection[0][0] + rand_r * cos(rand_theta)) / g_allParams.mcwidth);
				l = (int)((g_allParams.LocationInitialInfection[0][1] + rand_r * sin(rand_theta)) / g_allParams.mcheight);
				j = k * g_allParams.nmch + l;
				counter++;
			}
			if (counter < 100)
			{
				g_allParams.LocationInitialInfection[0][0] = g_allParams.LocationInitialInfection[0][0] + rand_r * cos(rand_theta); //set LocationInitialInfection to actual one used
				g_allParams.LocationInitialInfection[0][1] = g_allParams.LocationInitialInfection[0][1] + rand_r * sin(rand_theta);
			}
		}
		if (Mcells[j].n < g_allParams.NumInitialInfections[0])
			ERR_CRITICAL("Too few people in seed microcell to start epidemic with required number of initial infectionz.\n");
	}
	fprintf(stderr, "Checking cells...\n");
	maxd = ((double)g_allParams.populationSize);
	last_i = 0;
	for (i = 0; i < g_allParams.microcellCount; i++)
		if (Mcells[i].n > 0) last_i = i;
	fprintf(stderr, "Allocating place/age groups...\n");
	for (k = 0; k < NUM_AGE_GROUPS * AGE_GROUP_WIDTH; k++)
	{
		for (l = 0; l < g_allParams.PlaceTypeNum; l++)
		{
			PropPlaces[k][l] = PropPlacesC[k][l] = 0.0;
			if ((k < g_allParams.PlaceTypeAgeMax[l]) && (k >= g_allParams.PlaceTypeAgeMin[l]))
				PropPlaces[k][l] += g_allParams.PlaceTypePropAgeGroup[l];
			if ((k < g_allParams.PlaceTypeAgeMax2[l]) && (k >= g_allParams.PlaceTypeAgeMin2[l]))
				PropPlaces[k][l] += g_allParams.PlaceTypePropAgeGroup2[l];
			if ((k < g_allParams.PlaceTypeAgeMax3[l]) && (k >= g_allParams.PlaceTypeAgeMin3[l]))
				PropPlaces[k][l] += g_allParams.PlaceTypePropAgeGroup3[l];
			if (l == g_allParams.HotelPlaceType)
				PropPlacesC[k][l] = ((l > 0) ? PropPlacesC[k][l - 1] : 0);
			else
				PropPlacesC[k][l] = PropPlaces[k][l] + ((l > 0) ? PropPlacesC[k][l - 1] : 0);
		}
	}
	/*
		for(l=0;l<P.PlaceTypeNum;l++)
			{
			for(k=0;k<NUM_AGE_GROUPS*AGE_GROUP_WIDTH;k++)
				fprintf(stderr, "%i:%lg ",k,PropPlaces[k][l]);
			fprintf(stderr,"\n");
			}
	*/
	/*	if((P.DoAdUnits)&&(P.DoAdunitDemog))
			{for(i=0;i<P.NumAdunits;i++) free(State.InvAgeDist[i]);}
		else
			free(State.InvAgeDist[0]);
		free(State.InvAgeDist);
	*/	g_allParams.nsp = 0;
	if (g_allParams.DoPlaces)
		if (!(Places = (place * *)malloc(g_allParams.PlaceTypeNum * sizeof(place*)))) ERR_CRITICAL("Unable to allocate place storage\n");
	if ((g_allParams.DoSchoolFile) && (g_allParams.DoPlaces))
	{
		fprintf(stderr, "Reading school file\n");
		if (!(dat = fopen(SchoolFile, "rb"))) ERR_CRITICAL("Unable to open school file\n");
		fscanf(dat, "%i", &g_allParams.nsp);
		for (j = 0; j < g_allParams.nsp; j++)
		{
			fscanf(dat, "%i %i", &m, &(g_allParams.PlaceTypeMaxAgeRead[j]));
			if (!(Places[j] = (place*)calloc(m, sizeof(place)))) ERR_CRITICAL("Unable to allocate place storage\n");
			for (i = 0; i < m; i++)
				if (!(Places[j][i].AvailByAge = (unsigned short int*) malloc(g_allParams.PlaceTypeMaxAgeRead[j] * sizeof(unsigned short int)))) ERR_CRITICAL("Unable to allocate place storage\n");
			g_allParams.Nplace[j] = 0;
			for (i = 0; i < g_allParams.microcellCount; i++) Mcells[i].np[j] = 0;
		}
		mr = 0;
		while (!feof(dat))
		{
			fscanf(dat, "%lg %lg %i %i", &x, &y, &j, &m);
			for (i = 0; i < g_allParams.PlaceTypeMaxAgeRead[j]; i++) fscanf(dat, "%hu", &(Places[j][g_allParams.Nplace[j]].AvailByAge[i]));
			Places[j][g_allParams.Nplace[j]].loc_x = (float)(x - g_allParams.SpatialBoundingBox[0]);
			Places[j][g_allParams.Nplace[j]].loc_y = (float)(y - g_allParams.SpatialBoundingBox[1]);
			if ((x >= g_allParams.SpatialBoundingBox[0]) && (x < g_allParams.SpatialBoundingBox[2]) && (y >= g_allParams.SpatialBoundingBox[1]) && (y < g_allParams.SpatialBoundingBox[3]))
			{
				i = g_allParams.nch * ((int)(Places[j][g_allParams.Nplace[j]].loc_x / g_allParams.cwidth)) + ((int)(Places[j][g_allParams.Nplace[j]].loc_y / g_allParams.cheight));
				if (Cells[i].n == 0) mr++;
				Places[j][g_allParams.Nplace[j]].n = m;
				i = (int)(Places[j][g_allParams.Nplace[j]].loc_x / g_allParams.mcwidth);
				k = (int)(Places[j][g_allParams.Nplace[j]].loc_y / g_allParams.mcheight);
				j2 = i * g_allParams.nmch + k;
				Mcells[j2].np[j]++;
				Places[j][g_allParams.Nplace[j]].mcell = j2;
				g_allParams.Nplace[j]++;
				if (g_allParams.Nplace[j] % 1000 == 0) fprintf(stderr, "%i read    \r", g_allParams.Nplace[j]);
			}
		}
		fclose(dat);
		fprintf(stderr, "%i schools read (%i in empty cells)      \n", g_allParams.Nplace[j], mr);
		for (i = 0; i < g_allParams.microcellCount; i++)
			for (j = 0; j < g_allParams.nsp; j++)
				if (Mcells[i].np[j] > 0)
				{
					if (!(Mcells[i].places[j] = (int*)malloc(Mcells[i].np[j] * sizeof(int)))) ERR_CRITICAL("Unable to allocate place storage\n");
					Mcells[i].np[j] = 0;
				}
		for (j = 0; j < g_allParams.nsp; j++)
		{
			t = s = 0;
			for (i = 0; i < g_allParams.populationSize; i++)
				t += PropPlaces[HOST_AGE_YEAR(i)][j];
			for (i = 0; i < g_allParams.Nplace[j]; i++)
			{
				k = Places[j][i].mcell;
				Mcells[k].places[j][Mcells[k].np[j]++] = i;
				s += (double)Places[j][i].n;
			}
			fprintf(stderr, "School type %i: capacity=%lg demand=%lg\n", j, s, t);
			t /= s;
			for (i = 0; i < g_allParams.Nplace[j]; i++)
				Places[j][i].n = (int)ceil(((double)Places[j][i].n) * t);
		}
	}
	if (g_allParams.DoPlaces)
	{
		fprintf(stderr, "Configuring places...\n");
#pragma omp parallel for private(tn,j2,i,j,k,t,m,s,x,y,xh,yh) schedule(static,1)
		for (tn = 0; tn < g_allParams.NumThreads; tn++)
			for (j2 = g_allParams.nsp + tn; j2 < g_allParams.PlaceTypeNum; j2 += g_allParams.NumThreads)
			{
				t = 0;
				g_allParams.PlaceTypeMaxAgeRead[j2] = 0;
				for (i = 0; i < g_allParams.populationSize; i++)
					t += PropPlaces[HOST_AGE_YEAR(i)][j2];
				g_allParams.Nplace[j2] = (int)ceil(t / g_allParams.PlaceTypeMeanSize[j2]);
				fprintf(stderr, "[%i:%i %g] ", j2, g_allParams.Nplace[j2], t);
				if (!(Places[j2] = (place*)calloc(g_allParams.Nplace[j2], sizeof(place)))) ERR_CRITICAL("Unable to allocate place storage\n");
				t = 1.0;
				for (m = i = k = 0; i < g_allParams.microcellCount; i++)
				{
					s = ((double) Mcells[i].n) / maxd / t;
					if (s > 1.0) s = 1.0;
					if (i == last_i)
						m += (Mcells[last_i].np[j2] = g_allParams.Nplace[j2] - m);
					else
						m += (Mcells[i].np[j2] = (int)ignbin_mt((long)(g_allParams.Nplace[j2] - m), s, tn));
					t -= ((double)Mcells[i].n) / maxd;
					if (Mcells[i].np[j2] > 0)
					{
						if (!(Mcells[i].places[j2] = (int*)malloc(Mcells[i].np[j2] * sizeof(int)))) ERR_CRITICAL("Unable to allocate place storage\n");
						x = (double)(i / g_allParams.nmch);
						y = (double)(i % g_allParams.nmch);
						for (j = 0; j < Mcells[i].np[j2]; j++)
						{
							xh = g_allParams.mcwidth * (ranf_mt(tn) + x);
							yh = g_allParams.mcheight * (ranf_mt(tn) + y);
							Places[j2][k].loc_x = (float)xh;
							Places[j2][k].loc_y = (float)yh;
							Places[j2][k].n = 0;
							Places[j2][k].mcell = i;
							Places[j2][k].country = Mcells[i].country;
							Mcells[i].places[j2][j] = k;
							k++;
						}
					}
				}
			}
		for (k = 0; k < NUM_AGE_GROUPS * AGE_GROUP_WIDTH; k++)
			for (l = 1; l < g_allParams.PlaceTypeNum; l++)
				if (l != g_allParams.HotelPlaceType)
				{
					if (PropPlacesC[k][l - 1] < 1)
						PropPlaces[k][l] /= (1 - PropPlacesC[k][l - 1]);
					else if (PropPlaces[k][l] != 0)
						PropPlaces[k][l] = 1.0;
				}
/*		for (j2 = 0; j2 < P.PlaceTypeNum; j2++)
			for (i =0; i < P.NMC; i++)
				if ((Mcells[i].np[j2]>0) && (Mcells[i].n == 0))
					fprintf(stderr, "\n##~ %i %i %i \n", i, j2, Mcells[i].np[j2]);
*/		fprintf(stderr, "Places assigned\n");
	}
	l = 0;
	for (j = 0; j < g_allParams.cellCount; j++)
		if (l < Cells[j].n) l = Cells[j].n;
	if (!(SamplingQueue = (int**)malloc(g_allParams.NumThreads * sizeof(int*)))) ERR_CRITICAL("Unable to allocate state storage\n");
	g_allParams.InfQueuePeakLength = g_allParams.populationSize / g_allParams.NumThreads / INF_QUEUE_SCALE;
#pragma omp parallel for private(i,k) schedule(static,1)
	for (i = 0; i < g_allParams.NumThreads; i++)
	{
		if (!(SamplingQueue[i] = (int*)malloc(2 * (MAX_PLACE_SIZE + CACHE_LINE_SIZE) * sizeof(int)))) ERR_CRITICAL("Unable to allocate state storage\n");
		for (k = 0; k < g_allParams.NumThreads; k++)
			if (!(StateT[i].inf_queue[k] = (infection*)malloc(g_allParams.InfQueuePeakLength * sizeof(infection)))) ERR_CRITICAL("Unable to allocate state storage\n");
		if (!(StateT[i].cell_inf = (float*)malloc((l + 1) * sizeof(float)))) ERR_CRITICAL("Unable to allocate state storage\n");
	}

	//set up queues and storage for digital contact tracing
	if ((g_allParams.DoAdUnits) && (g_allParams.DoDigitalContactTracing))
	{
		for (i = 0; i < g_allParams.NumAdunits; i++)
		{
			//malloc or calloc for these?
			if (!(AdUnits[i].dct = (int*)malloc(AdUnits[i].n * sizeof(int)))) ERR_CRITICAL("Unable to allocate state storage\n");
		}
		for (i = 0; i < g_allParams.NumThreads; i++)
		{
			for (j = 0; j < g_allParams.NumAdunits; j++)
			{
				if (!(StateT[i].dct_queue[j] = (contactevent*)malloc(AdUnits[j].n * sizeof(contactevent)))) ERR_CRITICAL("Unable to allocate state storage\n");
			}
		}

	}

	//If outputting origin-destination matrix, set up storage for flow between admin units
	if ((g_allParams.DoAdUnits) && (g_allParams.DoOriginDestinationMatrix))
	{
		for (i = 0; i < g_allParams.NumAdunits; i++)
		{
			if (!(AdUnits[i].origin_dest = (double*)malloc(MAX_ADUNITS * sizeof(double)))) ERR_CRITICAL("Unable to allocate storage for origin destination matrix\n");
			for (j = 0; j < g_allParams.NumThreads; j++)
			{
				if (!(StateT[j].origin_dest[i] = (double*)calloc(MAX_ADUNITS, sizeof(double)))) ERR_CRITICAL("Unable to allocate state origin destination matrix storage\n");
			}
			//initialise to zero
			for (j = 0; j < g_allParams.NumAdunits; j++)
			{
				AdUnits[i].origin_dest[j] = 0.0;
			}
		}
	}

	for (i = 0; i < g_allParams.cellCount; i++)
	{
		Cells[i].cumTC = 0;
		Cells[i].S = Cells[i].n;
		Cells[i].L = Cells[i].I = 0;
	}
	fprintf(stderr, "Allocated cell and host memory\n");
	fprintf(stderr, "Assigned hosts to cells\n");

}
void SetupAirports(void)
{
	int i, j, k, l, m;
	double x, y, t, tmin;
	indexlist* base, *cur;

	fprintf(stderr, "Assigning airports to microcells\n");
  // Convince static analysers that values are set correctly:
  if (!(g_allParams.DoAirports && g_allParams.HotelPlaceType < g_allParams.PlaceTypeNum)) ERR_CRITICAL("DoAirports || HotelPlaceType not set\n");

	g_allParams.KernelType = g_allParams.airportKernelType;
	g_allParams.KernelScale = g_allParams.AirportKernelScale;
	g_allParams.KernelShape = g_allParams.AirportKernelShape;
	g_allParams.KernelP3 = g_allParams.AirportKernelP3;
	g_allParams.KernelP4 = g_allParams.AirportKernelP4;
	InitKernel(1, 1.0);
	if (!(Airports[0].DestMcells = (indexlist*)calloc(g_allParams.NMCP * NNA, sizeof(indexlist)))) ERR_CRITICAL("Unable to allocate airport storage\n");
	if (!(base = (indexlist*)calloc(g_allParams.NMCP * NNA, sizeof(indexlist)))) ERR_CRITICAL("Unable to allocate airport storage\n");
	for (i = 0; i < g_allParams.Nairports; i++) Airports[i].num_mcell = 0;
	cur = base;
	for (i = 0; i < g_allParams.microcellCount; i++)
		if (Mcells[i].n > 0)
		{
			Mcells[i].AirportList = cur;
			cur += NNA;
		}
#pragma omp parallel for private(i,j,k,l,x,y,t,tmin) schedule(static,10000)
	for (i = 0; i < g_allParams.microcellCount; i++)
		if (Mcells[i].n > 0)
		{
			if (i % 10000 == 0) fprintf(stderr, "\n%i           ", i);
			x = (((double)(i / g_allParams.nmch)) + 0.5) * g_allParams.mcwidth;
			y = (((double)(i % g_allParams.nmch)) + 0.5) * g_allParams.mcheight;
			k = l = 0;
			tmin = 1e20;
			for (j = 0; j < g_allParams.Nairports; j++)
				if (Airports[j].total_traffic > 0)
				{
					t = numKernel(dist2_raw(x, y, Airports[j].loc_x, Airports[j].loc_y)) * Airports[j].total_traffic;
					if (k < NNA)
					{
						Mcells[i].AirportList[k].id = j;
						Mcells[i].AirportList[k].prob = (float)t;
						if (t < tmin) { tmin = t; l = k; }
						k++;
					}
					else if (t > tmin)
					{
						Mcells[i].AirportList[l].id = j;
						Mcells[i].AirportList[l].prob = (float)t;
						tmin = 1e20;
						for (k = 0; k < NNA; k++)
							if (Mcells[i].AirportList[k].prob < tmin)
							{
								tmin = Mcells[i].AirportList[k].prob;
								l = k;
							}
					}
				}
			for (j = 0; j < NNA; j++)
				Airports[Mcells[i].AirportList[j].id].num_mcell++;
		}
	cur = Airports[0].DestMcells;
	fprintf(stderr, "Microcell airport lists collated.\n");
	for (i = 0; i < g_allParams.Nairports; i++)
	{
		Airports[i].DestMcells = cur;
		cur += Airports[i].num_mcell;
		Airports[i].num_mcell = 0;
	}
#pragma omp parallel for private(i,j,k,l,t,tmin) schedule(static,10000)
	for (i = 0; i < g_allParams.microcellCount; i++)
		if (Mcells[i].n > 0)
		{
			if (i % 10000 == 0) fprintf(stderr, "\n%i           ", i);
			t = 0;
			for (j = 0; j < NNA; j++)
			{
				t += Mcells[i].AirportList[j].prob;
				k = Mcells[i].AirportList[j].id;
#pragma omp critical (airport)
				l = (Airports[k].num_mcell++);
				Airports[k].DestMcells[l].id = i;
				Airports[k].DestMcells[l].prob = Mcells[i].AirportList[j].prob * ((float)Mcells[i].n);
			}
			tmin = 0;
			for (j = 0; j < NNA; j++)
			{
				Mcells[i].AirportList[j].prob = (float)(tmin + Mcells[i].AirportList[j].prob / t);
				tmin = Mcells[i].AirportList[j].prob;
			}
		}
	fprintf(stderr, "Airport microcell lists collated.\n");
	for (i = 0; i < g_allParams.Nairports; i++)
		if (Airports[i].total_traffic > 0)
		{
			for (j = 1; j < Airports[i].num_mcell; j++)
				Airports[i].DestMcells[j].prob += Airports[i].DestMcells[j - 1].prob;
			t = Airports[i].DestMcells[Airports[i].num_mcell - 1].prob;
			if (t == 0) t = 1.0;
			for (j = 0; j < Airports[i].num_mcell - 1; j++)
				Airports[i].DestMcells[j].prob = (float)(Airports[i].DestMcells[j].prob / t);
			if (Airports[i].num_mcell > 0) Airports[i].DestMcells[Airports[i].num_mcell - 1].prob = 1.0;
			for (j = l = 0; l <= 1024; l++)
			{
				t = ((double)l) / 1024.0;
				while (Airports[i].DestMcells[j].prob < t) j++;
				Airports[i].Inv_DestMcells[l] = j;
			}
			l = 0;
			for (j = 0; j < Airports[i].num_mcell; j++)
				l += Mcells[Airports[i].DestMcells[j].id].np[g_allParams.HotelPlaceType];
			if (l < 10)
			{
				fprintf(stderr, "(%i ", l);
				l = 0;
				for (j = 0; j < Airports[i].num_mcell; j++)
					l += Mcells[Airports[i].DestMcells[j].id].n;
				fprintf(stderr, "%i %i) ", Airports[i].num_mcell, l);
			}
		}
	fprintf(stderr, "\nInitialising hotel to airport lookup tables\n");
	free(base);
#pragma omp parallel for private(i,j,l,m,t,tmin) schedule(static,1)
	for (i = 0; i < g_allParams.Nairports; i++)
		if (Airports[i].total_traffic > 0)
		{
			m = (int)(Airports[i].total_traffic / HOTELS_PER_1000PASSENGER / 1000);
			if (m < MIN_HOTELS_PER_AIRPORT) m = MIN_HOTELS_PER_AIRPORT;
			fprintf(stderr, "\n%i    ", i);
			tmin = MAX_DIST_AIRPORT_TO_HOTEL * MAX_DIST_AIRPORT_TO_HOTEL * 0.75;
			do
			{
				tmin += 0.25 * MAX_DIST_AIRPORT_TO_HOTEL * MAX_DIST_AIRPORT_TO_HOTEL;
				Airports[i].num_place = 0;
				for (j = 0; j < g_allParams.Nplace[g_allParams.HotelPlaceType]; j++)
					if (dist2_raw(Airports[i].loc_x, Airports[i].loc_y,
						Places[g_allParams.HotelPlaceType][j].loc_x, Places[g_allParams.HotelPlaceType][j].loc_y) < tmin)
						Airports[i].num_place++;
			} while (Airports[i].num_place < m);
			if (tmin > MAX_DIST_AIRPORT_TO_HOTEL * MAX_DIST_AIRPORT_TO_HOTEL) fprintf(stderr, "*** %i : %lg %i ***\n", i, sqrt(tmin), Airports[i].num_place);
			if (!(Airports[i].DestPlaces = (indexlist*)calloc(Airports[i].num_place, sizeof(indexlist)))) ERR_CRITICAL("Unable to allocate airport storage\n");
			Airports[i].num_place = 0;
			for (j = 0; j < g_allParams.Nplace[g_allParams.HotelPlaceType]; j++)
				if ((t = dist2_raw(Airports[i].loc_x, Airports[i].loc_y,
					Places[g_allParams.HotelPlaceType][j].loc_x, Places[g_allParams.HotelPlaceType][j].loc_y)) < tmin)
				{
					Airports[i].DestPlaces[Airports[i].num_place].prob = (float)numKernel(t);
					Airports[i].DestPlaces[Airports[i].num_place].id = j;
					Airports[i].num_place++;
				}
			t = 0;
			for (j = 0; j < Airports[i].num_place; j++)
			{
				Airports[i].DestPlaces[j].prob = (float)(t + Airports[i].DestPlaces[j].prob);
				t = Airports[i].DestPlaces[j].prob;
			}
			for (j = 0; j < Airports[i].num_place - 1; j++)
				Airports[i].DestPlaces[j].prob = (float)(Airports[i].DestPlaces[j].prob / t);
			if (Airports[i].num_place > 0) Airports[i].DestPlaces[Airports[i].num_place - 1].prob = 1.0;
			for (j = l = 0; l <= 1024; l++)
			{
				t = ((double)l) / 1024.0;
				while (Airports[i].DestPlaces[j].prob < t) j++;
				Airports[i].Inv_DestPlaces[l] = j;
			}
		}
	g_allParams.KernelType = g_allParams.moveKernelType;
	g_allParams.KernelScale = g_allParams.MoveKernelScale;
	g_allParams.KernelShape = g_allParams.MoveKernelShape;
	g_allParams.KernelP3 = g_allParams.MoveKernelP3;
	g_allParams.KernelP4 = g_allParams.MoveKernelP4;
	for (i = 0; i < g_allParams.Nplace[g_allParams.HotelPlaceType]; i++) Places[g_allParams.HotelPlaceType][i].n = 0;
	InitKernel(0, 1.0);
	fprintf(stderr, "\nAirport initialisation completed successfully\n");
}

#define PROP_OTHER_PARENT_AWAY 0.0


void AssignHouseholdAges(int n, int pers, int tn)
{
	/* Complex household age distribution model
		- picks number of children (nc)
		- tries to space them reasonably
		- picks parental ages to be consistent with childrens' and each other
		- other adults in large households are assumed to be grandparents
		- for Thailand, 2 person households are 95% couples without children, 5% 1 parent families
	*/
	int i, j, k, l, nc, ad;
	int a[MAX_HOUSEHOLD_SIZE + 2];

	ad = ((g_allParams.DoAdunitDemog) && (g_allParams.DoAdUnits)) ? Mcells[Hosts[pers].mcell].adunit : 0;
	if (!g_allParams.DoHouseholds)
	{
		for (i = 0; i < n; i++)
			a[i] = State.InvAgeDist[ad][(int)(1000.0 * ranf_mt(tn))];
	}
	else
	{
		if (n == 1)
		{
			if (ranf_mt(tn) < g_allParams.OnePersHouseProbOld)
			{
				do
				{
					a[0] = State.InvAgeDist[ad][(int)(1000.0 * ranf_mt(tn))];
				}
				while ((a[0] < g_allParams.NoChildPersAge)
					|| (ranf_mt(tn) > (((double)a[0]) - g_allParams.NoChildPersAge + 1) / (g_allParams.OldPersAge - g_allParams.NoChildPersAge + 1)));
			}
			else if ((g_allParams.OnePersHouseProbYoung > 0) && (ranf_mt(tn) < g_allParams.OnePersHouseProbYoung / (1 - g_allParams.OnePersHouseProbOld)))
			{
				do
				{
					a[0] = State.InvAgeDist[ad][(int)(1000.0 * ranf_mt(tn))];
				} while ((a[0] > g_allParams.YoungAndSingle) || (a[0] < g_allParams.MinAdultAge)
					|| (ranf_mt(tn) > 1 - g_allParams.YoungAndSingleSlope * (((double)a[0]) - g_allParams.MinAdultAge) / (g_allParams.YoungAndSingle - g_allParams.MinAdultAge)));
			}
			else
				while ((a[0] = State.InvAgeDist[ad][(int)(1000.0 * ranf_mt(tn))]) < g_allParams.MinAdultAge);
		}
		else if (n == 2)
		{
			if (ranf_mt(tn) < g_allParams.TwoPersHouseProbOld)
			{
				do
				{
					a[0] = State.InvAgeDist[ad][(int)(1000.0 * ranf_mt(tn))];
				}
				while ((a[0] < g_allParams.NoChildPersAge)
					|| (ranf_mt(tn) > (((double)a[0]) - g_allParams.NoChildPersAge + 1) / (g_allParams.OldPersAge - g_allParams.NoChildPersAge + 1)));
				do
				{
					a[1] = State.InvAgeDist[ad][(int)(1000.0 * ranf_mt(tn))];
				}
				while ((a[1] > a[0] + g_allParams.MaxMFPartnerAgeGap) || (a[1] < a[0] - g_allParams.MaxFMPartnerAgeGap) || (a[1] < g_allParams.NoChildPersAge)
					|| (ranf_mt(tn) > (((double)a[1]) - g_allParams.NoChildPersAge + 1) / (g_allParams.OldPersAge - g_allParams.NoChildPersAge + 1)));
			}
			else if (ranf_mt(tn) < g_allParams.OneChildTwoPersProb / (1 - g_allParams.TwoPersHouseProbOld))
			{
				while ((a[0] = State.InvAgeDist[ad][(int)(1000.0 * ranf_mt(tn))]) > g_allParams.MaxChildAge);
				do
				{
					a[1] = State.InvAgeDist[ad][(int)(1000.0 * ranf_mt(tn))];
				}
				while ((a[1] > a[0] + g_allParams.MaxParentAgeGap) || (a[1] < a[0] + g_allParams.MinParentAgeGap) || (a[1] < g_allParams.MinAdultAge));
			}
			else if ((g_allParams.TwoPersHouseProbYoung > 0) && (ranf_mt(tn) < g_allParams.TwoPersHouseProbYoung / (1 - g_allParams.TwoPersHouseProbOld - g_allParams.OneChildTwoPersProb)))
			{
				do
				{
					a[0] = State.InvAgeDist[ad][(int)(1000.0 * ranf_mt(tn))];
				} while ((a[0] < g_allParams.MinAdultAge) || (a[0] > g_allParams.YoungAndSingle)
					|| (ranf_mt(tn) > 1 - g_allParams.YoungAndSingleSlope * (((double)a[0]) - g_allParams.MinAdultAge) / (g_allParams.YoungAndSingle - g_allParams.MinAdultAge)));
				do
				{
					a[1] = State.InvAgeDist[ad][(int)(1000.0 * ranf_mt(tn))];
				}
				while ((a[1] > a[0] + g_allParams.MaxMFPartnerAgeGap) || (a[1] < a[0] - g_allParams.MaxFMPartnerAgeGap) || (a[1] < g_allParams.MinAdultAge));
			}
			else
			{
				do
				{
					a[0] = State.InvAgeDist[ad][(int)(1000.0 * ranf_mt(tn))];
				} while (a[0] < g_allParams.MinAdultAge);
				do
				{
					a[1] = State.InvAgeDist[ad][(int)(1000.0 * ranf_mt(tn))];
				}
				while ((a[1] > a[0] + g_allParams.MaxMFPartnerAgeGap) || (a[1] < a[0] - g_allParams.MaxFMPartnerAgeGap) || (a[1] < g_allParams.MinAdultAge));
			}

		}
		else
		{
			if (n == 3)
			{
				if ((g_allParams.ZeroChildThreePersProb > 0) || (g_allParams.TwoChildThreePersProb > 0))
					nc = (ranf_mt(tn) < g_allParams.ZeroChildThreePersProb) ? 0 : ((ranf_mt(tn) < g_allParams.TwoChildThreePersProb) ? 2 : 1);
				else
					nc = 1;
			}
			else if (n == 4)
				nc = (ranf_mt(tn) < g_allParams.OneChildFourPersProb) ? 1 : 2;
			else if (n == 5)
				nc = (ranf_mt(tn) < g_allParams.ThreeChildFivePersProb) ? 3 : 2;
			else
				nc = n - 2 - (int)(3 * ranf_mt(tn));
			if (nc <= 0)
			{
				do
				{
					a[0] = State.InvAgeDist[ad][(int)(1000.0 * ranf_mt(tn))];
					a[1] = State.InvAgeDist[ad][(int)(1000.0 * ranf_mt(tn))];
				}
				while ((a[1] < g_allParams.MinAdultAge) || (a[0] < g_allParams.MinAdultAge));
				do
				{
					a[2] = State.InvAgeDist[ad][(int)(1000.0 * ranf_mt(tn))];
				}
				while ((a[2] >= a[1] + g_allParams.MaxMFPartnerAgeGap) || (a[2] < a[1] - g_allParams.MaxFMPartnerAgeGap));
			}
			else
			{
				do
				{
					a[0] = 0;
					for (i = 1; i < nc; i++)
						a[i] = a[i - 1] + 1 + ((int)ignpoi_mt(g_allParams.MeanChildAgeGap - 1, tn));
					a[0] = State.InvAgeDist[ad][(int)(1000.0 * ranf_mt(tn))] - a[(int)(ranf_mt(tn) * ((double)nc))];
					for (i = 1; i < nc; i++) a[i] += a[0];
					k = (((nc == 1) && (ranf_mt(tn) < g_allParams.OneChildProbYoungestChildUnderFive)) || ((nc == 2) && (ranf_mt(tn) < g_allParams.TwoChildrenProbYoungestUnderFive))
						|| ((nc > 2) && (ranf_mt(tn) < g_allParams.ProbYoungestChildUnderFive))) ? 5 : g_allParams.MaxChildAge;
				} while ((a[0] < 0) || (a[0] > k) || (a[nc - 1] > g_allParams.MaxChildAge));
				j = a[nc - 1] - a[0] - (g_allParams.MaxParentAgeGap - g_allParams.MinParentAgeGap);
				if (j > 0)
					j += g_allParams.MaxParentAgeGap;
				else
					j = g_allParams.MaxParentAgeGap;
				do
				{
					a[nc] = State.InvAgeDist[ad][(int)(1000.0 * ranf_mt(tn))];
					k = a[nc - 1];
					l = k - g_allParams.MaxChildAge;
				} while ((a[nc] > a[0] + j) || (a[nc] < k + g_allParams.MinParentAgeGap) || (a[nc] < g_allParams.MinAdultAge));
				if ((n > nc + 1) && (ranf_mt(tn) > PROP_OTHER_PARENT_AWAY))
				{
					do
					{
						a[nc + 1] = State.InvAgeDist[ad][(int)(1000.0 * ranf_mt(tn))];
					} while ((a[nc + 1] > a[nc] + g_allParams.MaxMFPartnerAgeGap) || (a[nc + 1] < a[nc] - g_allParams.MaxFMPartnerAgeGap)
						|| (a[nc + 1] > a[0] + j) || (a[nc + 1] < k + g_allParams.MinParentAgeGap) || (a[nc + 1] < g_allParams.MinAdultAge));
					l = nc + 2;
				}
				else
					l = nc + 1;
				if (n > nc + 2)
				{
					j = ((a[nc + 1] > a[nc]) ? a[nc + 1] : a[nc]) + g_allParams.OlderGenGap;
					if (j >= NUM_AGE_GROUPS * AGE_GROUP_WIDTH) j = NUM_AGE_GROUPS * AGE_GROUP_WIDTH - 1;
					if (j < g_allParams.NoChildPersAge) j = g_allParams.NoChildPersAge;
					for (i = nc + 2; i < n; i++)
						while ((a[i] = State.InvAgeDist[ad][(int)(1000.0 * ranf_mt(tn))]) < j);
				}
			}
		}
	}
	for (i = 0; i < n; i++) Hosts[pers + i].age = (unsigned char) a[i];
}

void AssignPeopleToPlaces(void)
{
	int i, i2, j, j2, k, k2, l, m, m2, tp, f, f2, f3, f4, ic, mx, my, a, cnt, tn, ca, nt, nn;
	int* PeopleArray;
	int* NearestPlaces[MAX_NUM_THREADS];
	double s, t, s2, *NearestPlacesProb[MAX_NUM_THREADS];
	cell* ct;
	int npt;

	npt = NUM_PLACE_TYPES;

	if (g_allParams.DoPlaces)
	{
		fprintf(stderr, "Assigning people to places....\n");
		for (i = 0; i < g_allParams.cellCount; i++)
		{
			Cells[i].infected = Cells[i].susceptible;
			if (!(Cells[i].susceptible = (int*)calloc(Cells[i].n, sizeof(int)))) ERR_CRITICAL("Unable to allocate state storage\n");
			Cells[i].cumTC = Cells[i].n;
		}

		//PropPlaces initialisation is only valid for non-overlapping places.

		for (i = 0; i < g_allParams.populationSize; i++)
		{
			for (tp = 0; tp < npt; tp++) //Changed from 'for(tp=0;tp<P.PlaceTypeNum;tp++)' to try and assign -1 early and avoid problems when using less than the default number of placetypes later
			{
				Hosts[i].PlaceLinks[tp] = -1;
			}
		}

		for (tp = 0; tp < g_allParams.PlaceTypeNum; tp++)
		{
			if (tp != g_allParams.HotelPlaceType)
			{
				cnt = 0;
				for (a = 0; a < g_allParams.populatedCellCount; a++)
				{
					cell *c = CellLookup[a];
					c->n = 0;
					for (j = 0; j < c->cumTC; j++)
					{
						k = HOST_AGE_YEAR(c->members[j]);
						f = ((PropPlaces[k][tp] > 0) && (ranf() < PropPlaces[k][tp]));
						if (f)
							for (k = 0; (k < tp) && (f); k++)
								if (Hosts[c->members[j]].PlaceLinks[k] >= 0) f = 0; //(ranf()<P.PlaceExclusivityMatrix[tp][k]);
						// Am assuming people can only belong to 1 place (and a hotel) at present
						if (f)
						{
							c->susceptible[c->n] = c->members[j];
							(c->n)++;
							cnt++;
						}
					}
					c->S = c->n;
					c->I = 0;
				}
				if (!(PeopleArray = (int*)calloc(cnt, sizeof(int)))) ERR_CRITICAL("Unable to allocate cell storage\n");
				j2 = 0;
				for (a = 0; a < g_allParams.populatedCellCount; a++)
				{
					cell *c = CellLookup[a];
					for (j = 0; j < c->n; j++)
					{
						PeopleArray[j2] = c->susceptible[j];
						j2++;
					}
				}
				// Use the Fisher–Yates shuffle algorithm to get a random permutation of PeopleArray
				for (int index1 = cnt - 1; index1 > 0; index1--)
				{
					int index2 = (int)(((double)k) * ranf());
					int tmp = PeopleArray[index1];
					PeopleArray[index1] = PeopleArray[index2];
					PeopleArray[index2] = tmp;
				}
				m = 0;
				if (tp < g_allParams.nsp)
				{
					for (i = 0; i < g_allParams.Nplace[tp]; i++)
					{
						m += (int)(Places[tp][i].treat_end_time = (unsigned short)Places[tp][i].n);
						Places[tp][i].n = 0;
					}
				}
				else if (g_allParams.PlaceTypeSizePower[tp] == 0 && g_allParams.PlaceTypeSizeSD[tp] == 0)
				{
					for (i = 0; i < g_allParams.Nplace[tp]; i++)
					{
						Places[tp][i].n = 0;
						j = 1 + ((int)ignpoi(g_allParams.PlaceTypeMeanSize[tp] - 1));
						if (j > USHRT_MAX - 1) j = USHRT_MAX - 1;
						m += (int)(Places[tp][i].treat_end_time = (unsigned short)j);
					}
				}
				//added this code to allow a place size to be specified according to a lognormal distribution - ggilani 09/02/17
				else if (g_allParams.PlaceTypeSizePower[tp] == 0 && g_allParams.PlaceTypeSizeSD[tp] > 0)
				{
					for (i = 0; i < g_allParams.Nplace[tp]; i++)
					{
						Places[tp][i].n = 0;
						j = (int)gen_lognormal(g_allParams.PlaceTypeMeanSize[tp], g_allParams.PlaceTypeSizeSD[tp]);
						if (j > USHRT_MAX - 1) j = USHRT_MAX - 1;
						m += (int)(Places[tp][i].treat_end_time = (unsigned short)j);
					}
				}
				else
				{
					s = pow(g_allParams.PlaceTypeSizeOffset[tp] / (g_allParams.PlaceTypeSizeOffset[tp] + g_allParams.PlaceTypeSizeMax[tp] - 1), g_allParams.PlaceTypeSizePower[tp]);
					for (i = 0; i < g_allParams.Nplace[tp]; i++)
					{
						j = (int)floor(g_allParams.PlaceTypeSizeOffset[tp] * pow((1 - s) * ranf() + s, -1 / g_allParams.PlaceTypeSizePower[tp]) + 1 - g_allParams.PlaceTypeSizeOffset[tp]);
						if (j > USHRT_MAX - 1) j = USHRT_MAX - 1;
						m += (int)(Places[tp][i].treat_end_time = (unsigned short)j);
						Places[tp][i].n = 0;
					}
				}
				if (tp < g_allParams.nsp)
				{
					t = ((double)m) / ((double)g_allParams.Nplace[tp]);
					fprintf(stderr, "Adjusting place weights by cell (Capacity=%i Demand=%i  Av place size=%lg)\n", m, cnt, t);
					for (i = 0; i < g_allParams.Nplace[tp]; i++)
						if (Places[tp][i].treat_end_time > 0)
						{
							j = (int)(Places[tp][i].loc_x / g_allParams.cwidth);
							k = j * g_allParams.nch + ((int)(Places[tp][i].loc_y / g_allParams.cheight));
							Cells[k].I += (int)Places[tp][i].treat_end_time;
						}
					for (k = 0; k < g_allParams.cellCount; k++)
					{
						i = k % g_allParams.nch;
						j = k / g_allParams.nch;
						f2 = Cells[k].I; f3 = Cells[k].S;
						if ((i > 0) && (j > 0))
						{
							f2 += Cells[(j - 1) * g_allParams.nch + (i - 1)].I; f3 += Cells[(j - 1) * g_allParams.nch + (i - 1)].S;
						}
						if (i > 0)
						{
							f2 += Cells[j * g_allParams.nch + (i - 1)].I; f3 += Cells[j * g_allParams.nch + (i - 1)].S;
						}
						if ((i > 0) && (j < g_allParams.ncw - 1))
						{
							f2 += Cells[(j + 1) * g_allParams.nch + (i - 1)].I; f3 += Cells[(j + 1) * g_allParams.nch + (i - 1)].S;
						}
						if (j > 0)
						{
							f2 += Cells[(j - 1) * g_allParams.nch + i].I; f3 += Cells[(j - 1) * g_allParams.nch + i].S;
						}
						if (j < g_allParams.ncw - 1)
						{
							f2 += Cells[(j + 1) * g_allParams.nch + i].I; f3 += Cells[(j + 1) * g_allParams.nch + i].S;
						}
						if ((i < g_allParams.nch - 1) && (j > 0))
						{
							f2 += Cells[(j - 1) * g_allParams.nch + (i + 1)].I; f3 += Cells[(j - 1) * g_allParams.nch + (i + 1)].S;
						}
						if (i < g_allParams.nch - 1)
						{
							f2 += Cells[j * g_allParams.nch + (i + 1)].I; f3 += Cells[j * g_allParams.nch + (i + 1)].S;
						}
						if ((i < g_allParams.nch - 1) && (j < g_allParams.ncw - 1))
						{
							f2 += Cells[(j + 1) * g_allParams.nch + (i + 1)].I; f3 += Cells[(j + 1) * g_allParams.nch + (i + 1)].S;
						}
						Cells[k].L = f3; Cells[k].R = f2;
					}
					m = f2 = f3 = f4 = 0;
					for (k = 0; k < g_allParams.cellCount; k++)
						if ((Cells[k].S > 0) && (Cells[k].I == 0))
						{
							f2 += Cells[k].S; f3++;
							if (Cells[k].R == 0) f4 += Cells[k].S;
						}
					fprintf(stderr, "Demand in cells with no places=%i in %i cells\nDemand in cells with no places <=1 cell away=%i\n", f2, f3, f4);
					for (i = 0; i < g_allParams.Nplace[tp]; i++)
						if (Places[tp][i].treat_end_time > 0)
						{
							j = (int)(Places[tp][i].loc_x / g_allParams.cwidth);
							k = j * g_allParams.nch + ((int)(Places[tp][i].loc_y / g_allParams.cheight));
							if ((Cells[k].L > 0) && (Cells[k].R > 0))
							{
								s = ((double)Cells[k].L) / ((double)Cells[k].R);
								Places[tp][i].treat_end_time = (unsigned short)ceil(Places[tp][i].treat_end_time * s);
							}
							m += ((int)Places[tp][i].treat_end_time);
						}
					for (i = 0; i < g_allParams.cellCount; i++) Cells[i].L = Cells[i].I = Cells[i].R = 0;
				}
				t = ((double)m) / ((double)g_allParams.Nplace[tp]);
				fprintf(stderr, "Adjusting place weights (Capacity=%i Demand=%i  Av place size=%lg)\n", m, cnt, t);
				for (i = m = 0; i < g_allParams.Nplace[tp]; i++)
				{
					s = ((double)Places[tp][i].treat_end_time) * 43 / 40 - 1;
					m += (int)(Places[tp][i].treat_end_time = (unsigned short)(1.0 + ignpoi(s)));
				}
				if (tp < g_allParams.nsp)
					s = ((double)cnt) * 1.075;
				else
					s = ((double)cnt) * 1.125;
				j2 = ((int)s) - m;
				for (i = 0; i < j2; i++)
				{
					Places[tp][(int)(((double)g_allParams.Nplace[tp]) * ranf())].treat_end_time++;
				}
				j2 = -j2;
				for (i = 0; i < j2; i++)
				{
					while (Places[tp][j = (int)(((double)g_allParams.Nplace[tp]) * ranf())].treat_end_time < 2);
					Places[tp][j].treat_end_time--;
				}
				if (g_allParams.PlaceTypeNearestNeighb[tp] == 0)
				{
					for (i = 0; i < g_allParams.cellCount; i++) Cells[i].S = 0;
					for (j = 0; j < g_allParams.Nplace[tp]; j++)
					{
						i = g_allParams.nch * ((int)(Places[tp][j].loc_x / g_allParams.cwidth)) + ((int)(Places[tp][j].loc_y / g_allParams.cheight));
						Cells[i].S += (int)Places[tp][j].treat_end_time;
					}
					for (i = 0; i < g_allParams.cellCount; i++)
					{
						if (Cells[i].S > Cells[i].cumTC)
						{
							free(Cells[i].susceptible);
							if (!(Cells[i].susceptible = (int*)calloc(Cells[i].S, sizeof(int)))) ERR_CRITICAL("Unable to allocate cell storage\n");
						}
						Cells[i].S = 0;
					}
					for (j = 0; j < g_allParams.Nplace[tp]; j++)
					{
						i = g_allParams.nch * ((int)(Places[tp][j].loc_x / g_allParams.cwidth)) + ((int)(Places[tp][j].loc_y / g_allParams.cheight));
						k = (int)Places[tp][j].treat_end_time;
						for (j2 = 0; j2 < k; j2++)
						{
							Cells[i].susceptible[Cells[i].S] = j;
							Cells[i].S++;
						}
					}
				}
				for (i = 0; i < g_allParams.NumThreads; i++)
				{
					if (!(NearestPlaces[i] = (int*)calloc(g_allParams.PlaceTypeNearestNeighb[tp] + CACHE_LINE_SIZE, sizeof(int)))) ERR_CRITICAL("Unable to allocate cell storage\n");
					if (!(NearestPlacesProb[i] = (double*)calloc(g_allParams.PlaceTypeNearestNeighb[tp] + CACHE_LINE_SIZE, sizeof(double)))) ERR_CRITICAL("Unable to allocate cell storage\n");
				}
				g_allParams.KernelType = g_allParams.PlaceTypeKernelType[tp];
				g_allParams.KernelScale = g_allParams.PlaceTypeKernelScale[tp];
				g_allParams.KernelShape = g_allParams.PlaceTypeKernelShape[tp];
				g_allParams.KernelP3 = g_allParams.PlaceTypeKernelP3[tp];
				g_allParams.KernelP4 = g_allParams.PlaceTypeKernelP4[tp];
				InitKernel(1, 1.0);
				UpdateProbs(1);
				ca = 0;
				fprintf(stderr, "Allocating people to place type %i\n", tp);
				a = cnt;
				nt = g_allParams.NumThreads;
				nn = g_allParams.PlaceTypeNearestNeighb[tp];
				if (g_allParams.PlaceTypeNearestNeighb[tp] > 0)
				{
#pragma omp parallel for private(i,i2,j,j2,k,k2,l,m,m2,f,f2,ic,cnt,tn,s,t,mx,my) firstprivate(a,nt,nn) reduction(+:ca) schedule(static,1) //added i3, nh_assigned to private
					for (tn = 0; tn < g_allParams.NumThreads; tn++)
					{
						for (j = tn; j < a; j += nt)
						{
							if (j % 1000 == 0) fprintf(stderr, "(%i) %i      \r", tp, j);
							for (i2 = 0; i2 < nn; i2++)
								NearestPlacesProb[tn][i2] = 0;
							l = 1; k = m = m2 = f2 = 0;
							i = PeopleArray[j];
							ic = Hosts[i].mcell;
							mx = ic / g_allParams.nmch;
							my = ic % g_allParams.nmch;
							if (Hosts[i].PlaceLinks[tp] < 0) //added this so that if any hosts have already be assigned due to their household membership, they will not be reassigned
							{
								while (((k < nn) || (l < 4)) && (l < g_allParams.nmcw))
								{
									if ((mx >= 0) && (my >= 0) && (mx < g_allParams.nmcw) && (my < g_allParams.nmch))
									{
										ic = mx * g_allParams.nmch + my;
										if (Mcells[ic].country == Mcells[Hosts[i].mcell].country)
										{
											for (cnt = 0; cnt < Mcells[ic].np[tp]; cnt++)
											{
												if (Mcells[ic].places[tp][cnt] >= g_allParams.Nplace[tp]) fprintf(stderr, "#%i %i %i  ", tp, ic, cnt);
												t = dist2_raw(Households[Hosts[i].hh].loc_x, Households[Hosts[i].hh].loc_y,
													Places[tp][Mcells[ic].places[tp][cnt]].loc_x, Places[tp][Mcells[ic].places[tp][cnt]].loc_y);
												s = numKernel(t);
												if (tp < g_allParams.nsp)
												{
													t = ((double)Places[tp][Mcells[ic].places[tp][cnt]].treat_end_time);
													if (HOST_AGE_YEAR(i) < g_allParams.PlaceTypeMaxAgeRead[tp])
													{
														if ((t > 0) && (Places[tp][Mcells[ic].places[tp][cnt]].AvailByAge[HOST_AGE_YEAR(i)] > 0))
															s *= t;
														else
															s = 0;
													}
													else if (t > 0)
														s *= t;
												}
												k2 = 0; j2 = 0; t = 1e10;
												if (s > 0)
												{
													if (k < nn)
													{
														NearestPlaces[tn][k] = Mcells[ic].places[tp][cnt];
														NearestPlacesProb[tn][k] = s;
														k++;
													}
													else
													{
														for (i2 = 0; i2 < nn; i2++)
														{
															if (NearestPlacesProb[tn][i2] < t)
															{
																t = NearestPlacesProb[tn][i2]; j2 = i2;
															}
														}
														if (s > t)
														{
															NearestPlacesProb[tn][j2] = s;
															NearestPlaces[tn][j2] = Mcells[ic].places[tp][cnt];
														}
													}
												}
											}
										}
									}
									if (m2 == 0)
										mx = mx + 1;
									else if (m2 == 1)
										my = my - 1;
									else if (m2 == 2)
										mx = mx - 1;
									else if (m2 == 3)
										my = my + 1;
									f2 = (f2 + 1) % l;
									if (f2 == 0)
									{
										m2 = (m2 + 1) % 4;
										m = (m + 1) % 2;
										if (m == 0) l++;
									}
								}

								s = 0;
								if (k > nn) fprintf(stderr, "*** k>P.PlaceTypeNearestNeighb[tp] ***\n");
								if (k == 0)
								{
									fprintf(stderr, "# %i %i     \r", i, j);
									Hosts[i].PlaceLinks[tp] = -1;
								}
								else
								{
									for (i2 = 1; i2 < k; i2++)
										NearestPlacesProb[tn][i2] += NearestPlacesProb[tn][i2 - 1];
									s = NearestPlacesProb[tn][k - 1];
									t = ranf_mt(tn);
									f = 0;
									for (i2 = 0; (i2 < k) && (!f); i2++)
									{
										if ((f = (t < NearestPlacesProb[tn][i2] / s)))
										{
											Hosts[i].PlaceLinks[tp] = NearestPlaces[tn][i2];
											ca++;
											if (tp < g_allParams.nsp)
											{
#pragma omp critical (places_treat_time)
												Places[tp][Hosts[i].PlaceLinks[tp]].treat_end_time--;
											}
										}
										if (!f) Hosts[i].PlaceLinks[tp] = -1;
										if (NearestPlaces[tn][i2] >= g_allParams.Nplace[tp]) fprintf(stderr, "@%i %i %i  ", tp, i, j);
									}
								}
							}
						}

					}
				}
				else
				{
					k2 = cnt - ca;
					m2 = cnt;
					s2 = 0;
					a = k2 / 1000;
					f = k2;
					for (ic = 0; ic <= 30; ic++)
					{
						UpdateProbs(1);
						m2 = f - 1;
						if (ic < 9)
							f = 100 * (9 - ic) * a;
						else if (ic < 18)
							f = 10 * (18 - ic) * a;
						else if (ic < 27)
							f = (27 - ic) * a;
						else
						{
							m2 = k2 - 1; f = 0;
						}
#pragma omp parallel for private(i,i2,j,k,l,m,f2,f3,t,ct,s) reduction(+:ca) /* schedule(dynamic,500)*/ //add s to private variables, added g,g1,g2,i3 and nh_assigned to private variables
						for (i2 = m2; i2 >= f; i2--)
						{
							if (i2 % 1000 == 0)
								fprintf(stderr, "(%i) %i            \r", tp, i2);
							k = PeopleArray[i2];
							i = Hosts[k].pcell;
							f2 = 1;
							f3 = (HOST_AGE_YEAR(k) >= g_allParams.PlaceTypeMaxAgeRead[tp]);
							if (Hosts[k].PlaceLinks[tp] < 0)
								while ((f2 > 0) && (f2 < 1000))
								{
									do
									{
										s = ranf();
										l = Cells[i].InvCDF[(int)floor(s * 1024)];
										while (Cells[i].cum_trans[l] < s) l++;
										ct = CellLookup[l];
										m = (int)(ranf() * ((double)ct->S));
										j = -1;
#pragma omp critical
										{
											if (ct->susceptible[m] >= 0)
												if ((f3) || (Places[tp][ct->susceptible[m]].AvailByAge[HOST_AGE_YEAR(k)] > 0))
												{
													j = ct->susceptible[m];
													ct->susceptible[m] = -1;
												}
										}
									} while (j < 0);
									if (j >= g_allParams.Nplace[tp])
									{
										fprintf(stderr, "*%i %i: %i %i\n", k, tp, j, g_allParams.Nplace[tp]);
										ERR_CRITICAL("Out of bounds place link\n");
									}
									t = dist2_raw(Households[Hosts[k].hh].loc_x, Households[Hosts[k].hh].loc_y, Places[tp][j].loc_x, Places[tp][j].loc_y);
									s = ((double)ct->S) / ((double)ct->S0) * numKernel(t) / Cells[i].max_trans[l];
									if ((g_allParams.DoAdUnits) && (g_allParams.InhibitInterAdunitPlaceAssignment[tp] > 0))
									{
										if (Mcells[Hosts[k].mcell].adunit != Mcells[Places[tp][j].mcell].adunit) s *= (1 - g_allParams.InhibitInterAdunitPlaceAssignment[tp]);
									}
									if (ranf() < s)
									{
#pragma omp critical
										l = (--ct->S);
										if (m < l) ct->susceptible[m] = ct->susceptible[l];
#pragma omp critical (places_treat_time)
										Places[tp][j].treat_end_time--;
										ca++;
										Hosts[k].PlaceLinks[tp] = j;
										f2 = 0;
									}
									else
									{
										ct->susceptible[m] = j;
										f2++;
									}
								}
						}
					}
				}
				fprintf(stderr, "%i hosts assigned to placetype %i\n", ca, tp);
				free(PeopleArray);
				for (i = 0; i < g_allParams.Nplace[tp]; i++)
				{
					Places[tp][i].treat_end_time = 0;
					Places[tp][i].n = 0;
				}
				for (i = 0; i < g_allParams.NumThreads; i++)
				{
					free(NearestPlacesProb[i]);
					free(NearestPlaces[i]);
				}
			}
		}
		for (i = 0; i < g_allParams.cellCount; i++)
		{
			Cells[i].n = Cells[i].cumTC;
			Cells[i].cumTC = 0;
			Cells[i].S = Cells[i].I = Cells[i].L = Cells[i].R = 0;
			free(Cells[i].susceptible);
			Cells[i].susceptible = Cells[i].infected;
		}
		g_allParams.KernelScale = g_allParams.MoveKernelScale;
		g_allParams.KernelShape = g_allParams.MoveKernelShape;
		g_allParams.KernelType = g_allParams.moveKernelType;
		g_allParams.KernelP3 = g_allParams.MoveKernelP3;
		g_allParams.KernelP4 = g_allParams.MoveKernelP4;
	}

}
void StratifyPlaces(void)
{
	int i, j, k, l, m, n, tn;
	double t;

	if (g_allParams.DoPlaces)
	{
		fprintf(stderr, "Initialising groups in places\n");
#pragma omp parallel for private(i,j) schedule(static,500)
		for (i = 0; i < g_allParams.populationSize; i++)
			for (j = 0; j < NUM_PLACE_TYPES; j++)
				Hosts[i].PlaceGroupLinks[j] = 0;
		for (j = 0; j < g_allParams.PlaceTypeNum; j++)
			for (i = 0; i < g_allParams.Nplace[j]; i++)
				Places[j][i].n = 0;
#pragma omp parallel for private(i,j,k,l,m,n,t,tn) schedule(static,1)
		for (tn = 0; tn < g_allParams.NumThreads; tn++)
			for (j = tn; j < g_allParams.PlaceTypeNum; j += g_allParams.NumThreads)
			{
				if (j == g_allParams.HotelPlaceType)
				{
					l = 2 * ((int)g_allParams.PlaceTypeMeanSize[j]);
					for (i = 0; i < g_allParams.Nplace[j]; i++)
					{
						if (!(Places[j][i].members = (int*)calloc(l, sizeof(int)))) ERR_CRITICAL("Unable to allocate place storage\n");
						Places[j][i].n = 0;
					}
				}
				else
				{
					for (i = 0; i < g_allParams.populationSize; i++)
					{
						if (Hosts[i].PlaceLinks[j] >= 0)
							Places[j][Hosts[i].PlaceLinks[j]].n++;
					}
					for (i = 0; i < g_allParams.Nplace[j]; i++)
					{
						if (Places[j][i].n > 0)
						{
							if (!(Places[j][i].members = (int*)calloc(Places[j][i].n, sizeof(int)))) ERR_CRITICAL("Unable to allocate place storage\n");
						}
						Places[j][i].n = 0;
					}
					for (i = 0; i < g_allParams.populationSize; i++)
					{
						k = Hosts[i].PlaceLinks[j];
						if (k >= 0)
						{
							Places[j][k].members[Places[j][k].n] = i;
							Places[j][k].n++;
						}
					}
					for (i = 0; i < g_allParams.Nplace[j]; i++)
						if (Places[j][i].n > 0)
						{
							t = ((double)Places[j][i].n) / g_allParams.PlaceTypeGroupSizeParam1[j] - 1.0;
							if (t < 0)
								Places[j][i].ng = 1;
							else
								Places[j][i].ng = 1 + (int)ignpoi_mt(t, tn);
							if (!(Places[j][i].group_start = (int*)calloc(Places[j][i].ng, sizeof(int)))) ERR_CRITICAL("Unable to allocate place storage\n");
							if (!(Places[j][i].group_size = (int*)calloc(Places[j][i].ng, sizeof(int)))) ERR_CRITICAL("Unable to allocate place storage\n");
							m = Places[j][i].n - Places[j][i].ng;
							for (k = l = 0; k < Places[j][i].ng; k++)
							{
								t = 1 / ((double)(Places[j][i].ng - k));
								Places[j][i].group_start[k] = l;
								Places[j][i].group_size[k] = 1 + ignbin_mt((long)m, t, tn);
								m -= (Places[j][i].group_size[k] - 1);
								l += Places[j][i].group_size[k];
							}
							for (k = 0; k < Places[j][i].n; k++)
							{
								l = (int)(((double)Places[j][i].n) * ranf_mt(tn));
								n = Places[j][i].members[l];
								Places[j][i].members[l] = Places[j][i].members[k];
								Places[j][i].members[k] = n;
							}
							for (k = l = 0; k < Places[j][i].ng; k++)
								for (m = 0; m < Places[j][i].group_size[k]; m++)
								{
									Hosts[Places[j][i].members[l]].PlaceGroupLinks[j] = k;
									l++;
								}
						}
				}
			}

#pragma omp parallel for private(i,j,k,l) schedule(static,1)
		for (i = 0; i < g_allParams.NumThreads; i++)
		{
			for (k = 0; k < g_allParams.PlaceTypeNum; k++)
			{
				if (g_allParams.DoPlaceGroupTreat)
				{
					l = 0;
					for (j = 0; j < g_allParams.Nplace[k]; j++)
						l += (int)Places[k][j].ng;
					if (!(StateT[i].p_queue[k] = (int*)calloc(l, sizeof(int)))) ERR_CRITICAL("Unable to allocate state storage\n");
					if (!(StateT[i].pg_queue[k] = (int*)calloc(l, sizeof(int)))) ERR_CRITICAL("Unable to allocate state storage\n");
				}
				else
				{
					if (!(StateT[i].p_queue[k] = (int*)calloc(g_allParams.Nplace[k], sizeof(int)))) ERR_CRITICAL("Unable to allocate state storage\n");
					if (!(StateT[i].pg_queue[k] = (int*)calloc(g_allParams.Nplace[k], sizeof(int)))) ERR_CRITICAL("Unable to allocate state storage\n");
				}
			}
		}
		fprintf(stderr, "Groups initialised\n");
		/*		s2=t2=0;
				for(j=0;j<P.PlaceTypeNum;j++)
					{
					t=s=0;
					for(i=0;i<P.Nplace[j];i++)
						if(Places[j][i].ng>0)
							{
							for(k=0;k<Places[j][i].ng;k++)
								t+=(double) Places[j][i].group_size[k];
							s+=(double) Places[j][i].ng;
							}
					s2+=s;
					t2+=t;
					fprintf(stderr,"Mean group size for place type %i = %lg\n",j,t/s);
					}
				t=0;
				for(i=0;i<P.N;i++)
					for(j=0;j<P.PlaceTypeNum;j++)
						if(Hosts[i].PlaceLinks[j]>=0)
							t+=(double) Places[j][Hosts[i].PlaceLinks[j]].group_size[Hosts[i].PlaceGroupLinks[j]];
				fprintf(stderr,"Overall mean group size = %lg (%lg)\n",t/((double) P.N),t2/s2);
		*/
	}
}
void LoadPeopleToPlaces(char* NetworkFile)
{
	int i, j, k, l, m, n, npt, i2;
	long s1, s2;
	FILE* dat;
	int fileversion;

	if (!(dat = fopen(NetworkFile, "rb"))) ERR_CRITICAL("Unable to open network file for loading\n");
	fread_big(&fileversion, sizeof(fileversion), 1, dat);
	if (fileversion != NETWORK_FILE_VERSION)
	{
		ERR_CRITICAL("Incompatible network file - please rebuild using '/S:'.\n");
	}

	npt = g_allParams.PlaceTypeNoAirNum;
	fread_big(&i, sizeof(int), 1, dat);
	fread_big(&j, sizeof(int), 1, dat);
	fread_big(&s1, sizeof(long), 1, dat);
	fread_big(&s2, sizeof(long), 1, dat);
	if (i != npt) ERR_CRITICAL("Number of place types does not match saved value\n");
	if (j != g_allParams.populationSize) ERR_CRITICAL("Population size does not match saved value\n");
	if ((s1 != g_allParams.setupSeed1) || (s2 != g_allParams.setupSeed2)) ERR_CRITICAL("Random number seeds do not match saved values\n");
	k = (g_allParams.populationSize + 999999) / 1000000;
	for (i = 0; i < g_allParams.populationSize; i++)
		for (j = 0; j < g_allParams.PlaceTypeNum; j++)
			Hosts[i].PlaceLinks[j] = -1;
	for (i = i2 = 0; i < k; i++)
	{
		l = (i < k - 1) ? 1000000 : (g_allParams.populationSize - 1000000 * (k - 1));
		fread_big(&netbuf, sizeof(int), npt * l, dat);
		for (j = 0; j < l; j++)
		{
			n = j * npt;
			for (m = 0; m < npt; m++)
			{
				Hosts[i2].PlaceLinks[m] = netbuf[n + m];
				if (Hosts[i2].PlaceLinks[m] >= g_allParams.Nplace[m])
				{
					fprintf(stderr, "*%i %i: %i %i\n", i2, m, Hosts[i2].PlaceLinks[m], g_allParams.Nplace[m]);
					ERR_CRITICAL("Out of bounds place link\n");
				}
			}
			i2++;
		}
		fprintf(stderr, "%i loaded            \r", i * 1000000 + l);
	}

	/*	for(i=0;i<P.N;i++)
			{
			if((i+1)%100000==0) fprintf(stderr,"%i loaded            \r",i+1);
			fread_big(&(Hosts[i].PlaceLinks[0]),sizeof(int),P.PlaceTypeNum,dat);
			}
	*/	fprintf(stderr, "\n");
	fclose(dat);
	g_allParams.KernelScale = g_allParams.MoveKernelScale;
	g_allParams.KernelShape = g_allParams.MoveKernelShape;
	g_allParams.KernelP3 = g_allParams.MoveKernelP3;
	g_allParams.KernelP4 = g_allParams.MoveKernelP4;
}
void SavePeopleToPlaces(char* NetworkFile)
{
	int i, j, npt;
	FILE* dat;
	int fileversion = NETWORK_FILE_VERSION;

	npt = g_allParams.PlaceTypeNoAirNum;
	if (!(dat = fopen(NetworkFile, "wb"))) ERR_CRITICAL("Unable to open network file for saving\n");
	fwrite_big(&fileversion, sizeof(fileversion), 1, dat);

	if (g_allParams.PlaceTypeNum > 0)
	{
		fwrite_big(&npt, sizeof(int), 1, dat);
		fwrite_big(&(g_allParams.populationSize), sizeof(int), 1, dat);
		fwrite_big(&g_allParams.setupSeed1, sizeof(long), 1, dat);
		fwrite_big(&g_allParams.setupSeed2, sizeof(long), 1, dat);
		for (i = 0; i < g_allParams.populationSize; i++)
		{
			if ((i + 1) % 100000 == 0) fprintf(stderr, "%i saved            \r", i + 1);
			/*			fwrite_big(&(Hosts[i].spatial_norm),sizeof(float),1,dat);
			*/			fwrite_big(&(Hosts[i].PlaceLinks[0]), sizeof(int), npt, dat);
			for (j = 0; j < npt; j++)
				if (Hosts[i].PlaceLinks[j] >= g_allParams.Nplace[j])
				{
					fprintf(stderr, "*%i %i: %i %i\n", i, j, Hosts[i].PlaceLinks[j], g_allParams.Nplace[j]);
					ERR_CRITICAL("Out of bounds place link\n");
				}
		}
	}

	fprintf(stderr, "\n");
	fflush(dat);
	fclose(dat);
}

void SaveAgeDistrib(void)
{
	int i;
	FILE* dat;
	char outname[1024];

	sprintf(outname, "%s.agedist.xls", OutFilePath);
	if (!(dat = fopen(outname, "wb"))) ERR_CRITICAL("Unable to open output file\n");
	if (g_allParams.DoDeath)
	{
		fprintf(dat, "age\tfreq\tlifeexpect\n");
		for (i = 0; i < NUM_AGE_GROUPS; i++)
			fprintf(dat, "%i\ta%.10f\t%.10f\n", i, AgeDist[i], AgeDist2[i]);
		fprintf(dat, "\np\tlife_expec\tage\n");
		for (i = 0; i <= 1000; i++)
			fprintf(dat, "%.10f\t%.10f\t%i\n", ((double)i) / 1000, g_allParams.InvLifeExpecDist[0][i], State.InvAgeDist[0][i]);
	}
	else
	{
		fprintf(dat, "age\tfreq\n");
		for (i = 0; i < NUM_AGE_GROUPS; i++)
			fprintf(dat, "%i\t%.10f\n", i, AgeDist[i]);
	}

	fclose(dat);
}
