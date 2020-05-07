/*
(c) 2004-20 Neil Ferguson, Imperial College London (neil.ferguson@imperial.ac.uk)
   All rights reserved. Copying and distribution prohibited without prior permission.
*/

#include <errno.h>
#include <stddef.h>

#include "CovidSim.h"
#include "binio.h"
#include "Rand.h"
#include "Error.h"
#include "Dist.h"
#include "Kernels.h"
#include "Bitmap.h"
#include "Model.h"
#include "Param.h"
#include "SetupModel.h"
#include "SharedFuncs.h"
#include "ModelMacros.h"
#include "InfStat.h"
#include "CalcInfSusc.h"
#include "Update.h"
#ifdef _OPENMP
#include <omp.h>
#endif // _OPENMP

void ReadParams(char* ParamFile, char* PreParamFile);
void ReadInterventions(char* IntFile);
int GetXMLNode(FILE* dat, const char* NodeName, const char* ParentName, char* Value, int ResetFilePos);
void ReadAirTravel(char*);
void InitModel(int);                        // adding run number as a parameter for event log: ggilani - 15/10/2014
void SeedInfection(double, int*, int, int); // adding run number as a parameter for event log: ggilani - 15/10/2014
int RunModel(int);                          // adding run number as a parameter for event log: ggilani - 15/10/2014
void TravelReturnSweep(double);
void TravelDepartSweep(double);
void InfectSweep(double, int);        // added int as argument to InfectSweep to record run number: ggilani - 15/10/14
void IncubRecoverySweep(double, int); // added int as argument to record run number: ggilani - 15/10/14
int TreatSweep(double);
// void HospitalSweep(double); //added hospital sweep function: ggilani - 10/11/14
void DigitalContactTracingSweep(double); // added function to update contact tracing number
void SaveDistribs(void);
void SaveOriginDestMatrix(void); // added function to save origin destination matrix so it can be done separately to the
                                 // main results: ggilani - 13/02/15
void SaveResults(void);
void SaveSummaryResults(void);
void SaveRandomSeeds(void); // added this function to save random seeds for each run: ggilani - 09/03/17
void SaveEvents(void);      // added this function to save infection events from all realisations: ggilani - 15/10/14
void LoadSnapshot(void);
void SaveSnapshot(void);
void RecordInfTypes(void);
void RecordSample(double, int);

void CalcOriginDestMatrix_adunit(void); // added function to calculate origin destination matrix: ggilani 28/01/15

int GetInputParameter(FILE*, FILE*, const char*, const char*, void*, int, int, int);
int GetInputParameter2(FILE*, FILE*, const char*, const char*, void*, int, int, int);
int GetInputParameter3(FILE*, const char*, const char*, void*, int, int, int);

///// ***** ///// ***** ///// ***** ///// ***** ///// ***** ///// ***** ///// ***** ///// ***** ///// ***** ///// *****
//////// ***** ///// ***** /////
///// ***** ///// ***** ///// ***** ///// ***** ///// ***** GLOBAL VARIABLES (some structures in CovidSim.h file and
///some containers) - memory allocated later.
///// ***** ///// ***** ///// ***** ///// ***** ///// ***** ///// ***** ///// ***** ///// ***** ///// ***** ///// *****
//////// ***** ///// ***** /////

param g_allParams;
person* Hosts;
household* Households;
popvar State, StateT[MAX_NUM_THREADS];
cell* Cells;       // Cells[i] is the i'th cell
cell** CellLookup; // CellLookup[i] is a pointer to the i'th populated cell
microcell *Mcells, **McellLookup;
place** Places;
adminunit AdUnits[MAX_ADUNITS];
//// Time Series defs:
//// TimeSeries is an array of type results, used to store (unsurprisingly) a time series of every quantity in results.
///Mostly used in RecordSample. / TSMeanNE and TSVarNE are the mean and variance of non-extinct time series. TSMeanE and
///TSVarE are the mean and variance of extinct time series. TSMean and TSVar are pointers that point to either extinct
///or non-extinct.
results *TimeSeries, *TSMean, *TSVar, *TSMeanNE, *TSVarNE, *TSMeanE,
   *TSVarE; //// TimeSeries used in RecordSample, RecordInfTypes, SaveResults. TSMean and TSVar
airport* Airports;
bitmap_header* bmh;
// added declaration of pointer to events log: ggilani - 10/10/2014
events* InfEventLog;
int* nEvents;

double inftype[INFECT_TYPE_MASK], inftype_av[INFECT_TYPE_MASK], infcountry[MAX_COUNTRIES], infcountry_av[MAX_COUNTRIES],
   infcountry_num[MAX_COUNTRIES];
double indivR0[MAX_SEC_REC][MAX_GEN_REC], indivR0_av[MAX_SEC_REC][MAX_GEN_REC];
double inf_household[MAX_HOUSEHOLD_SIZE + 1][MAX_HOUSEHOLD_SIZE + 1], denom_household[MAX_HOUSEHOLD_SIZE + 1];
double inf_household_av[MAX_HOUSEHOLD_SIZE + 1][MAX_HOUSEHOLD_SIZE + 1], AgeDist[NUM_AGE_GROUPS],
   AgeDist2[NUM_AGE_GROUPS];
double case_household[MAX_HOUSEHOLD_SIZE + 1][MAX_HOUSEHOLD_SIZE + 1],
   case_household_av[MAX_HOUSEHOLD_SIZE + 1][MAX_HOUSEHOLD_SIZE + 1];
double PropPlaces[NUM_AGE_GROUPS * AGE_GROUP_WIDTH][NUM_PLACE_TYPES];
double PropPlacesC[NUM_AGE_GROUPS * AGE_GROUP_WIDTH][NUM_PLACE_TYPES], AirTravelDist[MAX_DIST];
double PeakHeightSum, PeakHeightSS, PeakTimeSum, PeakTimeSS;

// These allow up to about 2 billion people per pixel, which should be ample.
int32_t* bmPopulation; // The population in each bitmap pixel. Special value -1 means "country boundary"
int32_t* bmInfected;   // The number of infected people in each bitmap pixel.
int32_t* bmRecovered;  // The number of recovered people in each bitmap pixel.
int32_t* bmTreated;    // The number of treated people in each bitmap pixel.

char OutFilePath[1024]      = {};
char OutFileBasePath[1024]  = {};
char OutDensFile[1024]      = {};
char SnapshotLoadFile[1024] = {};
char SnapshotSaveFile[1024] = {};
char AdunitFile[1024]       = {};

int ns, DoInitUpdateProbs, InterruptRun = 0;
int PlaceDistDistrib[NUM_PLACE_TYPES][MAX_DIST], PlaceSizeDistrib[NUM_PLACE_TYPES][MAX_PLACE_SIZE];

/* int NumPC,NumPCD; */
#define MAXINTFILE 10

int main(int argc, char* argv[])
{
   char ParamFilePath[1024]                    = {};
   char DensityFilePath[1024]                  = {};
   char NetworkFilePath[1024]                  = {};
   char AirTravelFilePath[1024]                = {};
   char SchoolFilePath[1024]                   = {};
   char RegDemogFilePath[1024]                 = {};
   char InterventionFilePath[MAXINTFILE][1024] = {};
   char PreParamFilePath[1024]                 = {};
   char buf[2048]                              = {};
   char* sep                                   = nullptr;

   ///// Flags to ensure various parameters have been read; set to false as default.
   bool haveParamFilePath       = false;
   bool havePreparamFilePath    = false;
   bool haveOutputFileBasePath  = false;
   bool haveLoadNetworkFilePath = false;
   bool haveSaveNetworkFilePath = false;
   bool haveAirTravelFilePath   = false;
   bool haveSchoolFilePath      = false;
   int Perr                     = 0;

   fprintf(stderr,
           "sizeof(int)=%i sizeof(long)=%i sizeof(float)=%i sizeof(double)=%i sizeof(unsigned short int)=%i sizeof(int "
           "*)=%i\n",
           (int)sizeof(int), (int)sizeof(long), (int)sizeof(float), (int)sizeof(double),
           (int)sizeof(unsigned short int), (int)sizeof(int*));

   clock_t clockStartTime = clock();

   ///// Read in command line arguments - lots of things, e.g. random number seeds; (pre)parameter files; binary files;
   ///population data; output directory? etc.
   if (argc < 7)
   {
      Perr = 1;
   }
   else
   {
      ///// Get seeds.
      int seedArgPos = argc - 4;

      sscanf(argv[seedArgPos++], "%li", &g_allParams.setupSeed1);
      sscanf(argv[seedArgPos++], "%li", &g_allParams.setupSeed2);
      sscanf(argv[seedArgPos++], "%li", &g_allParams.runSeed1);
      sscanf(argv[seedArgPos++], "%li", &g_allParams.runSeed2);

      ///// Set parameter defaults - read them in after
      g_allParams.PlaceCloseIndepThresh            = 0;
      g_allParams.LoadSaveNetwork                  = 0;
      g_allParams.DoHeteroDensity                  = 0;
      g_allParams.DoPeriodicBoundaries             = 0;
      g_allParams.DoSchoolFile                     = 0;
      g_allParams.DoAdunitDemog                    = 0;
      g_allParams.OutputDensFile                   = 0;
      g_allParams.MaxNumThreads                    = 0;
      g_allParams.DoInterventionFile               = 0;
      g_allParams.PreControlClusterIdCaseThreshold = 0;
      g_allParams.R0scale                          = 1.0;
      g_allParams.KernelOffsetScale                = 1.0;
      g_allParams.KernelPowerScale                 = 1.0; // added this so that kernel parameters are only changed if input from the
                                          // command line: ggilani - 15/10/2014
      g_allParams.DoSaveSnapshot = 0;
      g_allParams.DoLoadSnapshot = 0;

      //// scroll through command line arguments, anticipating what they can be using various if statements.
      for (int i = 1; i < argc - 4; i++)
      {
         if ((argv[i][0] != '/') && ((argv[i][2] != ':') && (argv[i][3] != ':')))
         {
            Perr = 1;
         }

         if (argv[i][1] == 'P' && argv[i][2] == ':')
         {
            haveParamFilePath = 1;
            sscanf(&argv[i][3], "%s", ParamFilePath);
         }
         else if (argv[i][1] == 'O' && argv[i][2] == ':')
         {
            haveOutputFileBasePath = 1;
            sscanf(&argv[i][3], "%s", OutFileBasePath);
         }
         else if (argv[i][1] == 'D' && argv[i][2] == ':')
         {
            sscanf(&argv[i][3], "%s", DensityFilePath);
            g_allParams.DoHeteroDensity      = 1;
            g_allParams.DoPeriodicBoundaries = 0;
         }
         else if (argv[i][1] == 'A' && argv[i][2] == ':')
         {
            sscanf(&argv[i][3], "%s", AdunitFile);
         }
         else if (argv[i][1] == 'L' && argv[i][2] == ':')
         {
            haveLoadNetworkFilePath     = 1;
            g_allParams.LoadSaveNetwork = 1;
            sscanf(&argv[i][3], "%s", NetworkFilePath);
         }
         else if (argv[i][1] == 'S' && argv[i][2] == ':')
         {
            g_allParams.LoadSaveNetwork = 2;
            haveSaveNetworkFilePath     = 1;
            sscanf(&argv[i][3], "%s", NetworkFilePath);
         }
         else if (argv[i][1] == 'R' && argv[i][2] == ':')
         {
            sscanf(&argv[i][3], "%lf", &g_allParams.R0scale);
         }
         else if (argv[i][1] == 'K' && argv[i][2] == 'P'
                  && argv[i][3] == ':') // added Kernel Power and Offset scaling so that it can easily be altered from
                                        // the command line in order to vary the kernel quickly: ggilani - 15/10/14
         {
            sscanf(&argv[i][4], "%lf", &g_allParams.KernelPowerScale);
         }
         else if (argv[i][1] == 'K' && argv[i][2] == 'O' && argv[i][3] == ':')
         {
            sscanf(&argv[i][4], "%lf", &g_allParams.KernelOffsetScale);
         }
         else if (argv[i][1] == 'C' && argv[i][2] == 'L' && argv[i][3] == 'P' && argv[i][4] == '1'
                  && argv[i][5] == ':') // generic command line specified param - matched to #1 in param file
         {
            sscanf(&argv[i][6], "%lf", &g_allParams.clP1);
         }
         else if (argv[i][1] == 'C' && argv[i][2] == 'L' && argv[i][3] == 'P' && argv[i][4] == '2'
                  && argv[i][5] == ':') // generic command line specified param - matched to #2 in param file
         {
            sscanf(&argv[i][6], "%lf", &g_allParams.clP2);
         }
         else if (argv[i][1] == 'C' && argv[i][2] == 'L' && argv[i][3] == 'P' && argv[i][4] == '3'
                  && argv[i][5] == ':') // generic command line specified param - matched to #3 in param file
         {
            sscanf(&argv[i][6], "%lf", &g_allParams.clP3);
         }
         else if (argv[i][1] == 'C' && argv[i][2] == 'L' && argv[i][3] == 'P' && argv[i][4] == '4'
                  && argv[i][5] == ':') // generic command line specified param - matched to #4 in param file
         {
            sscanf(&argv[i][6], "%lf", &g_allParams.clP4);
         }
         else if (argv[i][1] == 'C' && argv[i][2] == 'L' && argv[i][3] == 'P' && argv[i][4] == '5'
                  && argv[i][5] == ':') // generic command line specified param - matched to #5 in param file
         {
            sscanf(&argv[i][6], "%lf", &g_allParams.clP5);
         }
         else if (argv[i][1] == 'C' && argv[i][2] == 'L' && argv[i][3] == 'P' && argv[i][4] == '6'
                  && argv[i][5] == ':') // generic command line specified param - matched to #6 in param file
         {
            sscanf(&argv[i][6], "%lf", &g_allParams.clP6);
         }
         else if (argv[i][1] == 'A' && argv[i][2] == 'P' && argv[i][3] == ':')
         {
            haveAirTravelFilePath = 1;
            sscanf(&argv[i][3], "%s", AirTravelFilePath);
         }
         else if (argv[i][1] == 's' && argv[i][2] == ':')
         {
            haveSchoolFilePath = 1;
            sscanf(&argv[i][3], "%s", SchoolFilePath);
         }
         else if (argv[i][1] == 'T' && argv[i][2] == ':')
         {
            sscanf(&argv[i][3], "%i", &g_allParams.PreControlClusterIdCaseThreshold);
         }
         else if (argv[i][1] == 'C' && argv[i][2] == ':')
         {
            sscanf(&argv[i][3], "%i", &g_allParams.PlaceCloseIndepThresh);
         }
         else if (argv[i][1] == 'd' && argv[i][2] == ':')
         {
            g_allParams.DoAdunitDemog = 1;
            sscanf(&argv[i][3], "%s", RegDemogFilePath);
         }
         else if (argv[i][1] == 'c' && argv[i][2] == ':')
         {
            sscanf(&argv[i][3], "%i", &g_allParams.MaxNumThreads);
         }
         else if (argv[i][1] == 'M' && argv[i][2] == ':')
         {
            g_allParams.OutputDensFile = 1;
            sscanf(&argv[i][3], "%s", OutDensFile);
         }
         else if (argv[i][1] == 'I' && argv[i][2] == ':')
         {
            sscanf(&argv[i][3], "%s", InterventionFilePath[g_allParams.DoInterventionFile]);
            g_allParams.DoInterventionFile++;
         }
         else if (argv[i][1] == 'L' && argv[i][2] == 'S' && argv[i][3] == ':')
         {
            sscanf(&argv[i][4], "%s", SnapshotLoadFile);
            g_allParams.DoLoadSnapshot = 1;
         }
         else if (argv[i][1] == 'P' && argv[i][2] == 'P' && argv[i][3] == ':')
         {
            sscanf(&argv[i][4], "%s", PreParamFilePath);
            havePreparamFilePath = 1;
         }
         else if (argv[i][1] == 'S' && argv[i][2] == 'S' && argv[i][3] == ':')
         {
            sscanf(&argv[i][4], "%s", buf);
            fprintf(stderr, "### %s\n", buf);
            sep = strchr(buf, ',');
            if (!sep)
               Perr = 1;
            else
            {
               g_allParams.DoSaveSnapshot = 1;
               *sep                       = ' ';
               sscanf(buf, "%lf %s", &(g_allParams.SnapshotSaveTime), SnapshotSaveFile);
            }
         }
      }
      if (((haveSaveNetworkFilePath) && (haveLoadNetworkFilePath)) || (!haveParamFilePath) || (!haveOutputFileBasePath))
      {
         Perr = 1;
      }
   }

   ///// END Read in command line arguments

   sprintf(OutFilePath, "%s", OutFileBasePath);

   fprintf(stderr, "Param=%s\nOut=%s\nDens=%s\n", ParamFilePath, OutFilePath, DensityFilePath);
   if (Perr)
   {
      ERR_CRITICAL_FMT(
         "Syntax:\n%s /P:ParamFilePath /O:OutputFile [/AP:AirTravelFilePath] [/s:SchoolFilePath] [/D:DensityFilePath] "
         "[/L:NetworkFileToLoad | /S:NetworkFileToSave] [/R:R0scaling] SetupSeed1 SetupSeed2 RunSeed1 RunSeed2\n",
         argv[0]);
   }

   //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** ////
   ///**** //// **** //// **** / **** SET UP OMP / THREADS / **** //// **** //// **** //// **** //// **** //// **** ////
   ///**** //// **** //// **** //// **** //// **** //// **** //// **** //// ****

#ifdef _OPENMP
   g_allParams.NumThreads = omp_get_max_threads();
   if ((g_allParams.MaxNumThreads > 0) && (g_allParams.MaxNumThreads < g_allParams.NumThreads))
      g_allParams.NumThreads = g_allParams.MaxNumThreads;
   if (g_allParams.NumThreads > MAX_NUM_THREADS)
   {
      fprintf(stderr, "Assigned number of threads (%d) > MAX_NUM_THREADS (%d)\n", g_allParams.NumThreads,
              MAX_NUM_THREADS);
      g_allParams.NumThreads = MAX_NUM_THREADS;
   }
   fprintf(stderr, "Using %d threads\n", g_allParams.NumThreads);
   omp_set_num_threads(g_allParams.NumThreads);

#pragma omp parallel default(shared)
   {
      fprintf(stderr, "Thread %i initialised\n", omp_get_thread_num());
   }
   /* fprintf(stderr,"int=%i\tfloat=%i\tdouble=%i\tint *=%i\n",(int) sizeof(int),(int) sizeof(float),(int)
    * sizeof(double),(int) sizeof(int *));	*/
#else
   g_allParams.NumThreads = 1;
#endif
   if (!havePreparamFilePath)
   {
      sprintf(PreParamFilePath, ".." DIRECTORY_SEPARATOR "Pre_%s", ParamFilePath);
   }

   //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** ////
   ///**** //// **** //// **** / **** READ IN PARAMETERS, DATA ETC. / **** //// **** //// **** //// **** //// **** ////
   ///**** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// ****

   ReadParams(ParamFilePath, PreParamFilePath);
   if (haveSchoolFilePath)
   {
      g_allParams.DoSchoolFile = 1;
   }

   if (g_allParams.DoAirports)
   {
      if (!haveAirTravelFilePath)
      {
         ERR_CRITICAL_FMT(
            "Syntax:\n%s /P:ParamFilePath /O:OutputFile /AP:AirTravelFilePath [/s:SchoolFilePath] [/D:DensityFilePath] "
            "[/L:NetworkFileToLoad | /S:NetworkFileToSave] [/R:R0scaling] SetupSeed1 SetupSeed2 RunSeed1 RunSeed2\n",
            argv[0]);
      }

      ReadAirTravel(AirTravelFilePath);
   }

   //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** ////
   ///**** //// **** //// **** / **** INITIALIZE / **** //// **** //// **** //// **** //// **** //// **** //// **** ////
   ///**** //// **** //// **** //// **** //// **** //// **** //// ****

   ///// initialize model (for all realisations).
   SetupModel(DensityFilePath, NetworkFilePath, SchoolFilePath, RegDemogFilePath);

   for (int i = 0; i < MAX_ADUNITS; i++)
   {
      AdUnits[i].NI = 0;
   }

   // g_allParams.DoInterventionFile can be zero.
   for (int i = 0; i < g_allParams.DoInterventionFile; ++i)
   {
      ReadInterventions(InterventionFilePath[i]);
   }

   fprintf(stderr, "Model setup in %lf seconds\n", (static_cast<double>(clock() - clockStartTime)) / CLOCKS_PER_SEC);

   // print out number of calls to random number generator

   //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** ////
   ///**** //// **** //// **** / **** RUN MODEL / **** //// **** //// **** //// **** //// **** //// **** //// **** ////
   ///**** //// **** //// **** //// **** //// **** //// **** //// ****

   g_allParams.NRactE  = 0;
   g_allParams.NRactNE = 0;
   for (int i = 0; (i < g_allParams.realisationNumber)
                   && (g_allParams.NRactNE < g_allParams.nonextinctRealisationNumber) && (!InterruptRun);
        ++i)
   {
      if (g_allParams.realisationNumber > 1)
      {
         sprintf(OutFilePath, "%s.%i", OutFileBasePath, i);
         fprintf(stderr, "Realisation %i   (time=%lf nr_ne=%i)\n", i + 1,
                 (static_cast<double>(clock() - clockStartTime)) / CLOCKS_PER_SEC, g_allParams.NRactNE);
      }

      ///// Set and save seeds
      if (i == 0 || (g_allParams.ResetSeeds && g_allParams.KeepSameSeeds))
      {
         g_allParams.nextRunSeed1 = g_allParams.runSeed1;
         g_allParams.nextRunSeed2 = g_allParams.runSeed2;
      }

      if (g_allParams.ResetSeeds)
      {
         // save these seeds to file
         SaveRandomSeeds();
      }

      // Now that we have set g_allParams.nextRunSeed* ready for the run, we need to save the values in case
      // we need to reinitialise the RNG after the run is interrupted.
      long thisRunSeed1 = g_allParams.nextRunSeed1;
      long thisRunSeed2 = g_allParams.nextRunSeed2;
      if (i == 0 || g_allParams.ResetSeeds)
      {
         setall(&g_allParams.nextRunSeed1, &g_allParams.nextRunSeed2);
         // fprintf(stderr, "%i, %i\n", g_allParams.newseed1,g_allParams.newseed2);
         // fprintf(stderr, "%f\n", ranf());
      }

      ///// initialize model (for this realisation).
      InitModel(i); // passing run number into RunModel so we can save run number in the infection event log: ggilani -
                    // 15/10/2014
      if (g_allParams.DoLoadSnapshot)
      {
         LoadSnapshot();
      }

      while (RunModel(i))
      { // has been interrupted to reset holiday time. Note that this currently only happens in the first run,
        // regardless of how many realisations are being run.
         long tmp1 = thisRunSeed1;
         long tmp2 = thisRunSeed2;
         setall(&tmp1, &tmp2); // reset random number seeds to generate same run again after calibration.
         InitModel(i);
      }

      if (g_allParams.OutputNonSummaryResults)
      {
         if (((!TimeSeries[g_allParams.totalSampleNumber - 1].extinct) || (!g_allParams.OutputOnlyNonExtinct))
             && (g_allParams.OutputEveryRealisation))
         {
            SaveResults();
         }
      }
      if ((g_allParams.DoRecordInfEvents) && (g_allParams.RecordInfEventsPerRun == 1))
      {
         SaveEvents();
      }
   }

   sprintf(OutFilePath, "%s", OutFileBasePath);

   // Calculate origin destination matrix if needed
   if (g_allParams.DoAdUnits && g_allParams.DoOriginDestinationMatrix)
   {
      CalcOriginDestMatrix_adunit();
      SaveOriginDestMatrix();
   }

   g_allParams.NRactual = g_allParams.NRactNE;
   TSMean               = TSMeanNE;
   TSVar                = TSVarNE;
   if (g_allParams.DoRecordInfEvents && g_allParams.RecordInfEventsPerRun == 0)
   {
      SaveEvents();
   }

   sprintf(OutFilePath, "%s.avNE", OutFileBasePath);
   SaveSummaryResults();
   g_allParams.NRactual = g_allParams.NRactE;
   TSMean               = TSMeanE;
   TSVar                = TSVarE;
   sprintf(OutFilePath, "%s.avE", OutFileBasePath);
   // SaveSummaryResults();

#ifdef WIN32_BM
   Gdiplus::GdiplusShutdown(m_gdiplusToken);
#endif
   fprintf(stderr, "Extinction in %i out of %i runs\n", g_allParams.NRactE, g_allParams.NRactNE + g_allParams.NRactE);
   fprintf(stderr, "Model ran in %lf seconds\n", (static_cast<double>(clock() - clockStartTime)) / CLOCKS_PER_SEC);
   fprintf(stderr, "Model finished\n");
}

void ReadParams(char* ParamFile, char* PreParamFile)
{
   FILE* ParamFile_dat    = nullptr;
   FILE* PreParamFile_dat = nullptr;
   FILE* AdminFile_dat    = nullptr;

   double s            = 0.0;
   double t            = 0.0;
   double AgeSuscScale = 1.0;

   int k  = 0;
   int f  = 0;
   int nc = 0;
   int na = 0;

   char CountryNameBuf[128 * MAX_COUNTRIES]   = {};
   char AdunitListNamesBuf[128 * MAX_ADUNITS] = {};

   char* CountryNames[MAX_COUNTRIES] = {};
   for (int i = 0; i < MAX_COUNTRIES; i++)
   {
      CountryNames[i]    = CountryNameBuf + 128 * i;
      CountryNames[i][0] = 0;
   }

   char* AdunitListNames[MAX_ADUNITS] = {};

   for (int i = 0; i < MAX_ADUNITS; i++)
   {
      AdunitListNames[i]    = AdunitListNamesBuf + 128 * i;
      AdunitListNames[i][0] = 0;
   }

   if (!(ParamFile_dat = fopen(ParamFile, "rb")))
   {
      ERR_CRITICAL("Unable to open parameter file\n");
   }

   PreParamFile_dat = fopen(PreParamFile, "rb");
   if (!(AdminFile_dat = fopen(AdunitFile, "rb")))
   {
      AdminFile_dat = ParamFile_dat;
   }

   AgeSuscScale = 1.0;
   GetInputParameter(ParamFile_dat, PreParamFile_dat, "Update timestep", "%lf", (void*)&(g_allParams.TimeStep), 1, 1,
                     0);
   GetInputParameter(ParamFile_dat, PreParamFile_dat, "Sampling timestep", "%lf", (void*)&(g_allParams.SampleStep), 1,
                     1, 0);

   if (g_allParams.TimeStep > g_allParams.SampleStep)
   {
      ERR_CRITICAL("Update step must be smaller than sampling step\n");
   }

   t                            = ceil(g_allParams.SampleStep / g_allParams.TimeStep - 1e-6);
   g_allParams.UpdatesPerSample = static_cast<int>(t);
   g_allParams.TimeStep         = g_allParams.SampleStep / t;
   g_allParams.TimeStepsPerDay  = ceil(1.0 / g_allParams.TimeStep - 1e-6);

   fprintf(stderr, "Update step = %lf\nSampling step = %lf\nUpdates per sample=%i\nTimeStepsPerDay=%lf\n",
           g_allParams.TimeStep, g_allParams.SampleStep, g_allParams.UpdatesPerSample, g_allParams.TimeStepsPerDay);

   GetInputParameter(ParamFile_dat, PreParamFile_dat, "Sampling time", "%lf", (void*)&(g_allParams.SampleTime), 1, 1,
                     0);

   g_allParams.totalSampleNumber = 1 + static_cast<int>(ceil(g_allParams.SampleTime / g_allParams.SampleStep));

   GetInputParameter(PreParamFile_dat, AdminFile_dat, "Population size", "%i", (void*)&(g_allParams.populationSize), 1,
                     1, 0);
   GetInputParameter(ParamFile_dat, PreParamFile_dat, "Number of realisations", "%i",
                     (void*)&(g_allParams.realisationNumber), 1, 1, 0);

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Number of non-extinct realisations", "%i",
                           reinterpret_cast<void*>(&(g_allParams.nonextinctRealisationNumber)), 1, 1, 0))
   {
      g_allParams.nonextinctRealisationNumber = g_allParams.realisationNumber;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Maximum number of cases defining small outbreak", "%i",
                           reinterpret_cast<void*>(&(g_allParams.SmallEpidemicCases)), 1, 1, 0))
   {
      g_allParams.SmallEpidemicCases = -1;
   }

   g_allParams.cellCount = -1;
   GetInputParameter(ParamFile_dat, PreParamFile_dat, "Number of micro-cells per spatial cell width", "%i",
                     reinterpret_cast<void*>(&(g_allParams.microcellsOnACellSide)), 1, 1, 0);

   // added parameter to reset seeds after every run
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Reset seeds for every run", "%i",
                           reinterpret_cast<void*>(&(g_allParams.ResetSeeds)), 1, 1, 0))
   {
      g_allParams.ResetSeeds = 0;
   }

   if (g_allParams.ResetSeeds)
   {
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Keep same seeds for every run", "%i",
                              reinterpret_cast<void*>(&(g_allParams.KeepSameSeeds)), 1, 1, 0))
      {
         g_allParams.KeepSameSeeds = 0; // added this to control which seeds are used: ggilani 27/11/19
      }
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Reset seeds after intervention", "%i",
                           reinterpret_cast<void*>(&(g_allParams.ResetSeedsPostIntervention)), 1, 1, 0))
   {
      g_allParams.ResetSeedsPostIntervention = 0;
   }

   if (g_allParams.ResetSeedsPostIntervention)
   {
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Time to reset seeds after intervention", "%i",
                              reinterpret_cast<void*>(&(g_allParams.TimeToResetSeeds)), 1, 1, 0))
      {
         g_allParams.TimeToResetSeeds = 1000000;
      }
   }

   if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Include households", "%i",
                           reinterpret_cast<void*>(&(g_allParams.DoHouseholds)), 1, 1, 0))
   {
      g_allParams.DoHouseholds = 1;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "OutputAge", "%i",
                           reinterpret_cast<void*>(&(g_allParams.OutputAge)), 1, 1, 0))
      g_allParams.OutputAge = 1; //// ON  by default.
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "OutputSeverityAdminUnit", "%i",
                           reinterpret_cast<void*>(&(g_allParams.OutputSeverityAdminUnit)), 1, 1, 0))
      g_allParams.OutputSeverityAdminUnit = 1; //// ON  by default.
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "OutputR0", "%i", (void*)&(g_allParams.OutputR0), 1, 1, 0))
      g_allParams.OutputR0 = 0; //// OFF by default.
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "OutputControls", "%i",
                           (void*)&(g_allParams.OutputControls), 1, 1, 0))
      g_allParams.OutputControls = 0; //// OFF by default.
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "OutputCountry", "%i", (void*)&(g_allParams.OutputCountry),
                           1, 1, 0))
      g_allParams.OutputCountry = 0; //// OFF by default.
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "OutputAdUnitVar", "%i",
                           (void*)&(g_allParams.OutputAdUnitVar), 1, 1, 0))
      g_allParams.OutputAdUnitVar = 0; //// OFF by default.
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "OutputHousehold", "%i",
                           (void*)&(g_allParams.OutputHousehold), 1, 1, 0))
      g_allParams.OutputHousehold = 0; //// OFF by default.
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "OutputInfType", "%i", (void*)&(g_allParams.OutputInfType),
                           1, 1, 0))
      g_allParams.OutputInfType = 0; //// OFF by default.
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "OutputNonSeverity", "%i",
                           (void*)&(g_allParams.OutputNonSeverity), 1, 1, 0))
      g_allParams.OutputNonSeverity = 0; //// OFF by default.
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "OutputNonSummaryResults", "%i",
                           (void*)&(g_allParams.OutputNonSummaryResults), 1, 1, 0))
      g_allParams.OutputNonSummaryResults = 0; //// OFF by default.

   if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Kernel resolution", "%i",
                           (void*)&g_allParams.kernelLookupTableSize, 1, 1, 0))
      g_allParams.kernelLookupTableSize = 4000000;
   if (g_allParams.kernelLookupTableSize < 2000000)
   {
      ERR_CRITICAL_FMT("[Kernel resolution] needs to be at least 2000000 - not %d", g_allParams.kernelLookupTableSize);
   }

   if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Kernel higher resolution factor", "%i",
                           (void*)&g_allParams.hiResKernelExpansionFactor, 1, 1, 0))
   {
      g_allParams.hiResKernelExpansionFactor = g_allParams.kernelLookupTableSize / 1600;
   }

   if (g_allParams.hiResKernelExpansionFactor < 1
       || g_allParams.hiResKernelExpansionFactor >= g_allParams.kernelLookupTableSize)
   {
      ERR_CRITICAL_FMT("[Kernel higher resolution factor] needs to be in range [1, g_allParams.NKR = %d) - not %d",
                       g_allParams.kernelLookupTableSize, g_allParams.hiResKernelExpansionFactor);
   }

   if (g_allParams.DoHouseholds)
   {
      GetInputParameter(PreParamFile_dat, AdminFile_dat, "Household size distribution", "%lf",
                        (void*)g_allParams.HouseholdSizeDistrib[0], MAX_HOUSEHOLD_SIZE, 1, 0);
      GetInputParameter(ParamFile_dat, PreParamFile_dat, "Household attack rate", "%lf",
                        (void*)&(g_allParams.HouseholdTrans), 1, 1, 0);
      GetInputParameter(ParamFile_dat, PreParamFile_dat, "Household transmission denominator power", "%lf",
                        (void*)&(g_allParams.HouseholdTransPow), 1, 1, 0);
      if (!GetInputParameter2(
             PreParamFile_dat, AdminFile_dat,
             "Correct age distribution after household allocation to exactly match specified demography", "%i",
             (void*)&(g_allParams.DoCorrectAgeDist), 1, 1, 0))
      {
         g_allParams.DoCorrectAgeDist = 0;
      }
   }
   else
   {
      g_allParams.HouseholdTrans             = 0.0;
      g_allParams.HouseholdTransPow          = 1.0;
      g_allParams.HouseholdSizeDistrib[0][0] = 1.0;
      for (int i = 1; i < MAX_HOUSEHOLD_SIZE; ++i)
      {
         g_allParams.HouseholdSizeDistrib[0][i] = 0;
      }
   }

   for (int i = 1; i < MAX_HOUSEHOLD_SIZE; i++)
   {
      g_allParams.HouseholdSizeDistrib[0][i] =
         g_allParams.HouseholdSizeDistrib[0][i] + g_allParams.HouseholdSizeDistrib[0][i - 1];
   }

   for (int i = 0; i < MAX_HOUSEHOLD_SIZE; i++)
   {
      g_allParams.HouseholdDenomLookup[i] = 1 / pow(((double)(i + 1)), g_allParams.HouseholdTransPow);
   }

   if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Include administrative units within countries", "%i",
                           (void*)&(g_allParams.DoAdUnits), 1, 1, 0))
   {
      g_allParams.DoAdUnits = 1;
   }

   if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Divisor for countries", "%i",
                           (void*)&(g_allParams.CountryDivisor), 1, 1, 0))
   {
      g_allParams.CountryDivisor = 1;
   }

   if (g_allParams.DoAdUnits)
   {
      char** AdunitNames   = nullptr;
      char* AdunitNamesBuf = nullptr;

      if (!(AdunitNames = (char**)malloc(3 * ADUNIT_LOOKUP_SIZE * sizeof(char*))))
      {
         ERR_CRITICAL("Unable to allocate temp storage\n");
      }

      if (!(AdunitNamesBuf = (char*)malloc(3 * ADUNIT_LOOKUP_SIZE * 360 * sizeof(char))))
      {
         ERR_CRITICAL("Unable to allocate temp storage\n");
      }

      for (int i = 0; i < ADUNIT_LOOKUP_SIZE; i++)
      {
         g_allParams.AdunitLevel1Lookup[i] = -1;
         AdunitNames[3 * i]                = AdunitNamesBuf + 3 * i * 360;
         AdunitNames[3 * i + 1]            = AdunitNamesBuf + 3 * i * 360 + 60;
         AdunitNames[3 * i + 2]            = AdunitNamesBuf + 3 * i * 360 + 160;
      }

      if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Divisor for level 1 administrative units", "%i",
                              (void*)&(g_allParams.AdunitLevel1Divisor), 1, 1, 0))
      {
         g_allParams.AdunitLevel1Divisor = 1;
      }

      if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Mask for level 1 administrative units", "%i",
                              (void*)&(g_allParams.AdunitLevel1Mask), 1, 1, 0))
      {
         g_allParams.AdunitLevel1Mask = 1000000000;
      }

      na = (GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Codes and country/province names for admin units",
                               "%s", (void*)AdunitNames, 3 * ADUNIT_LOOKUP_SIZE, 1, 0))
           / 3;
      if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Number of countries to include", "%i", (void*)&nc, 1, 1,
                              0))
      {
         nc = 0;
      }

      if ((na > 0) && (nc > 0))
      {
         g_allParams.DoAdunitBoundaries = (nc > 0);
         nc                             = abs(nc);
         GetInputParameter(PreParamFile_dat, AdminFile_dat, "List of names of countries to include", "%s",
                           (nc > 1) ? ((void*)CountryNames) : ((void*)CountryNames[0]), nc, 1, 0);
         g_allParams.NumAdunits = 0;
         for (int i = 0; i < na; ++i)
            for (int j = 0; j < nc; ++j)
               if ((AdunitNames[3 * i + 1][0]) && (!strcmp(AdunitNames[3 * i + 1], CountryNames[j]))
                   && (atoi(AdunitNames[3 * i]) != 0))
               {
                  AdUnits[g_allParams.NumAdunits].id                                = atoi(AdunitNames[3 * i]);
                  g_allParams.AdunitLevel1Lookup[(AdUnits[g_allParams.NumAdunits].id % g_allParams.AdunitLevel1Mask)
                                                 / g_allParams.AdunitLevel1Divisor] = g_allParams.NumAdunits;
                  if (strlen(AdunitNames[3 * i + 1]) < 100)
                  {
                     strcpy(AdUnits[g_allParams.NumAdunits].cnt_name, AdunitNames[3 * i + 1]);
                  }

                  if (strlen(AdunitNames[3 * i + 2]) < 200)
                  {
                     strcpy(AdUnits[g_allParams.NumAdunits].ad_name, AdunitNames[3 * i + 2]);
                  }

                  //						fprintf(stderr,"%i %s %s ##
                  //",AdUnits[g_allParams.NumAdunits].id,AdUnits[g_allParams.NumAdunits].cnt_name,AdUnits[g_allParams.NumAdunits].ad_name);
                  g_allParams.NumAdunits++;
               }
      }
      else
      {
         if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Number of level 1 administrative units to include",
                                 "%i", (void*)&(g_allParams.NumAdunits), 1, 1, 0))
         {
            g_allParams.NumAdunits = 0;
         }

         if (g_allParams.NumAdunits > 0)
         {
            g_allParams.DoAdunitBoundaries = 1;
            if (g_allParams.NumAdunits > MAX_ADUNITS)
            {
               ERR_CRITICAL("MAX_ADUNITS too small.\n");
            }

            GetInputParameter(PreParamFile_dat, AdminFile_dat, "List of level 1 administrative units to include", "%s",
                              (g_allParams.NumAdunits > 1) ? ((void*)AdunitListNames) : ((void*)AdunitListNames[0]),
                              g_allParams.NumAdunits, 1, 0);
            na = g_allParams.NumAdunits;
            for (int i = 0; i < g_allParams.NumAdunits; i++)
            {
               f = 0;
               if (na > 0)
               {
                  int j = 0;
                  for (; (j < na) && (!f); j++)
                  {
                     f = (!strcmp(AdunitNames[3 * j + 2], AdunitListNames[i]));
                  }

                  if (f)
                  {
                     k = atoi(AdunitNames[3 * (j - 1)]);
                  }
               }
               if ((na == 0) || (!f))
               {
                  k = atoi(AdunitListNames[i]);
               }

               AdUnits[i].id = k;

               g_allParams.AdunitLevel1Lookup[(k % g_allParams.AdunitLevel1Mask) / g_allParams.AdunitLevel1Divisor] = i;
               for (int j = 0; j < na; ++j)
                  if (atoi(AdunitNames[3 * j]) == k)
                  {
                     if (strlen(AdunitNames[3 * j + 1]) < 100)
                        strcpy(AdUnits[i].cnt_name, AdunitNames[3 * j + 1]);
                     if (strlen(AdunitNames[3 * j + 2]) < 200)
                        strcpy(AdUnits[i].ad_name, AdunitNames[3 * j + 2]);
                     j = na;
                  }
            }
         }
         else
         {
            g_allParams.DoAdunitBoundaries = 0;
         }
      }

      free(AdunitNames);
      free(AdunitNamesBuf);

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Output incidence by administrative unit", "%i",
                              (void*)&(g_allParams.DoAdunitOutput), 1, 1, 0))
      {
         g_allParams.DoAdunitOutput = 0;
      }

      if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Draw administrative unit boundaries on maps", "%i",
                              (void*)&(g_allParams.DoAdunitBoundaryOutput), 1, 1, 0))
      {
         g_allParams.DoAdunitBoundaryOutput = 0;
      }

      if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Correct administrative unit populations", "%i",
                              (void*)&(g_allParams.DoCorrectAdunitPop), 1, 1, 0))
      {
         g_allParams.DoCorrectAdunitPop = 0;
      }

      if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Fix population size at specified value", "%i",
                              (void*)&(g_allParams.DoSpecifyPop), 1, 1, 0))
      {
         g_allParams.DoSpecifyPop = 0;
      }

      fprintf(stderr, "Using %i administrative units\n", g_allParams.NumAdunits);

      if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat,
                              "Divisor for administrative unit codes for boundary plotting on bitmaps", "%i",
                              (void*)&(g_allParams.AdunitBitmapDivisor), 1, 1, 0))
      {
         g_allParams.AdunitBitmapDivisor = 1;
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Only output household to place distance distribution for one administrative unit", "%i",
                              (void*)&(g_allParams.DoOutputPlaceDistForOneAdunit), 1, 1, 0))
      {
         g_allParams.DoOutputPlaceDistForOneAdunit = 0;
      }

      if (g_allParams.DoOutputPlaceDistForOneAdunit)
      {
         if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                                 "Administrative unit for which household to place distance distribution to be output",
                                 "%i", (void*)&(g_allParams.OutputPlaceDistAdunit), 1, 1, 0))
         {
            g_allParams.DoOutputPlaceDistForOneAdunit = 0;
         }
      }
   }
   else
   {
      g_allParams.DoAdunitBoundaries     = 0;
      g_allParams.DoAdunitBoundaryOutput = 0;
      g_allParams.DoAdunitOutput         = 0;
      g_allParams.DoCorrectAdunitPop     = 0;
      g_allParams.DoSpecifyPop           = 0;
      g_allParams.AdunitLevel1Divisor    = 1;
      g_allParams.AdunitLevel1Mask       = 1000000000;
      g_allParams.AdunitBitmapDivisor    = g_allParams.AdunitLevel1Divisor;
   }

   if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Include age", "%i", (void*)&(g_allParams.DoAge), 1, 1, 0))
   {
      g_allParams.DoAge = 1;
   }

   if (!g_allParams.DoAge)
   {
      for (int i = 0; i < NUM_AGE_GROUPS; ++i)
      {
         g_allParams.PropAgeGroup[0][i] = 1.0 / NUM_AGE_GROUPS;
      }

      for (int i = 0; i < NUM_AGE_GROUPS; ++i)
      {
         g_allParams.InitialImmunity[i]        = 0;
         g_allParams.AgeInfectiousness[i]      = 1;
         g_allParams.AgeSusceptibility[i]      = 1;
         g_allParams.RelativeSpatialContact[i] = 1.0;
         g_allParams.RelativeTravelRate[i]     = 1.0;
      }
   }
   else
   {
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Initial immunity acts as partial immunity", "%i",
                              (void*)&(g_allParams.DoPartialImmunity), 1, 1, 0))
      {
         g_allParams.DoPartialImmunity = 1;
      }

      if ((g_allParams.DoHouseholds) && (!g_allParams.DoPartialImmunity))
      {
         if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Initial immunity applied to all household members",
                                 "%i", (void*)&(g_allParams.DoWholeHouseholdImmunity), 1, 1, 0))
         {
            g_allParams.DoWholeHouseholdImmunity = 0;
         }
      }
      else
      {
         g_allParams.DoWholeHouseholdImmunity = 0;
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Initial immunity profile by age", "%lf",
                              (void*)g_allParams.InitialImmunity, NUM_AGE_GROUPS, 1, 0))
      {
         for (int i = 0; i < NUM_AGE_GROUPS; i++)
         {
            g_allParams.InitialImmunity[i] = 0;
         }
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Relative spatial contact rates by age", "%lf",
                              (void*)g_allParams.RelativeSpatialContact, NUM_AGE_GROUPS, 1, 0))
      {
         for (int i = 0; i < NUM_AGE_GROUPS; ++i)
         {
            g_allParams.RelativeSpatialContact[i] = 1;
         }
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Age-dependent infectiousness", "%lf",
                              (void*)g_allParams.AgeInfectiousness, NUM_AGE_GROUPS, 1, 0))
      {
         for (int i = 0; i < NUM_AGE_GROUPS; ++i)
         {
            g_allParams.AgeInfectiousness[i] = 1.0;
         }
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Age-dependent susceptibility", "%lf",
                              (void*)g_allParams.AgeSusceptibility, NUM_AGE_GROUPS, 1, 0))
      {
         for (int i = 0; i < NUM_AGE_GROUPS; ++i)
         {
            g_allParams.AgeSusceptibility[i] = 1.0;
         }
      }

      GetInputParameter(PreParamFile_dat, AdminFile_dat, "Age distribution of population", "%lf",
                        (void*)g_allParams.PropAgeGroup[0], NUM_AGE_GROUPS, 1, 0);
      t = 0;
      for (int i = 0; i < NUM_AGE_GROUPS; ++i)
      {
         t += g_allParams.PropAgeGroup[0][i];
      }

      for (int i = 0; i < NUM_AGE_GROUPS; ++i)
      {
         g_allParams.PropAgeGroup[0][i] /= t;
      }

      t = 0;
      for (int i = 0; i < NUM_AGE_GROUPS; ++i)
      {
         if (g_allParams.AgeSusceptibility[i] > t)
            t = g_allParams.AgeSusceptibility[i]; // peak susc has to be 1
         {
            for (i = 0; i < NUM_AGE_GROUPS; i++)
            {
               g_allParams.AgeSusceptibility[i] /= t;
            }
         }
      }

      AgeSuscScale = t;
      if (g_allParams.DoHouseholds)
      {
         g_allParams.HouseholdTrans *= AgeSuscScale;
      }

      if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Relative travel rates by age", "%lf",
                              (void*)g_allParams.RelativeTravelRate, NUM_AGE_GROUPS, 1, 0))
      {
         for (int i = 0; i < NUM_AGE_GROUPS; ++i)
         {
            g_allParams.RelativeTravelRate[i] = 1;
         }
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "WAIFW matrix", "%lf", (void*)g_allParams.WAIFW_Matrix,
                              NUM_AGE_GROUPS, NUM_AGE_GROUPS, 0))
      {
         for (int i = 0; i < NUM_AGE_GROUPS; ++i)
         {
            for (int j = 0; j < NUM_AGE_GROUPS; ++j)
            {
               g_allParams.WAIFW_Matrix[i][j] = 1.0;
            }
         }
      }
      else
      {
         /* WAIFW matrix needs to be scaled to have max value of 1.
         1st index of matrix specifies host being infected, second the infector.
         Overall age variation in infectiousness/contact rates/susceptibility should be factored
         out of WAIFW_matrix and put in Age dep infectiousness/susceptibility for efficiency. */
         t = 0;
         for (int i = 0; i < NUM_AGE_GROUPS; ++i)
         {
            for (int j = 0; j < NUM_AGE_GROUPS; ++j)
            {
               if (g_allParams.WAIFW_Matrix[i][j] > t)
               {
                  t = g_allParams.WAIFW_Matrix[i][j];
               }
            }
         }

         if (t > 0)
         {
            for (int i = 0; i < NUM_AGE_GROUPS; ++i)
            {
               for (int j = 0; j < NUM_AGE_GROUPS; ++j)
               {
                  g_allParams.WAIFW_Matrix[i][j] /= t;
               }
            }
         }
         else
         {
            for (int i = 0; i < NUM_AGE_GROUPS; ++i)
            {
               for (int j = 0; j < NUM_AGE_GROUPS; ++j)
               {
                  g_allParams.WAIFW_Matrix[i][j] = 1.0;
               }
            }
         }
      }

      g_allParams.DoDeath = 0;
      t                   = 0;
      for (int i = 0; i < NUM_AGE_GROUPS; ++i)
      {
         t += g_allParams.AgeInfectiousness[i] * g_allParams.PropAgeGroup[0][i];
      }

      for (int i = 0; i < NUM_AGE_GROUPS; i++)
      {
         g_allParams.AgeInfectiousness[i] /= t;
      }
   }

   if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Include spatial transmission", "%i",
                           (void*)&(g_allParams.DoSpatial), 1, 1, 0))
   {
      g_allParams.DoSpatial = 1;
   }

   GetInputParameter(PreParamFile_dat, AdminFile_dat, "Kernel type", "%i", (void*)&(g_allParams.moveKernelType), 1, 1,
                     0);
   GetInputParameter(PreParamFile_dat, AdminFile_dat, "Kernel scale", "%lf", (void*)&(g_allParams.MoveKernelScale), 1,
                     1, 0);

   if (g_allParams.KernelOffsetScale != 1)
   {
      g_allParams.MoveKernelScale *= g_allParams.KernelOffsetScale;
   }

   if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Kernel 3rd param", "%lf",
                           (void*)&(g_allParams.MoveKernelP3), 1, 1, 0))
   {
      g_allParams.MoveKernelP3 = 0;
   }

   if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Kernel 4th param", "%lf",
                           (void*)&(g_allParams.MoveKernelP4), 1, 1, 0))
   {
      g_allParams.MoveKernelP4 = 0;
   }

   if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Kernel Shape", "%lf",
                           (void*)&(g_allParams.MoveKernelShape), 1, 1, 0))
   {
      g_allParams.MoveKernelShape = 1.0;
   }

   if (g_allParams.KernelPowerScale != 1)
   {
      g_allParams.MoveKernelShape *= g_allParams.KernelPowerScale;
   }

   if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Airport Kernel Type", "%i",
                           (void*)&(g_allParams.airportKernelType), 1, 1, 0))
   {
      g_allParams.airportKernelType = g_allParams.moveKernelType;
   }

   if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Airport Kernel Scale", "%lf",
                           (void*)&(g_allParams.AirportKernelScale), 1, 1, 0))
   {
      g_allParams.AirportKernelScale = g_allParams.MoveKernelScale;
   }

   if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Airport Kernel Shape", "%lf",
                           (void*)&(g_allParams.AirportKernelShape), 1, 1, 0))
   {
      g_allParams.AirportKernelShape = g_allParams.MoveKernelShape;
   }

   if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Airport Kernel 3rd param", "%lf",
                           (void*)&(g_allParams.AirportKernelP3), 1, 1, 0))
   {
      g_allParams.AirportKernelP3 = g_allParams.MoveKernelP3;
   }

   if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Airport Kernel 4th param", "%lf",
                           (void*)&(g_allParams.AirportKernelP4), 1, 1, 0))
   {
      g_allParams.AirportKernelP4 = g_allParams.MoveKernelP4;
   }

   if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Include places", "%i", (void*)&(g_allParams.DoPlaces), 1,
                           1, 0))
   {
      g_allParams.DoPlaces = 1;
   }

   if (g_allParams.DoPlaces)
   {
      if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Number of types of places", "%i",
                              (void*)&(g_allParams.PlaceTypeNum), 1, 1, 0))
      {
         g_allParams.PlaceTypeNum = 0;
      }

      if (g_allParams.PlaceTypeNum == 0)
      {
         g_allParams.DoPlaces   = 0;
         g_allParams.DoAirports = 0;
      }
   }
   else
   {
      g_allParams.PlaceTypeNum = 0;
      g_allParams.DoAirports   = 0;
   }

   if (g_allParams.DoPlaces)
   {
      if (g_allParams.PlaceTypeNum > NUM_PLACE_TYPES)
      {
         ERR_CRITICAL("Too many place types\n");
      }

      GetInputParameter(PreParamFile_dat, AdminFile_dat, "Minimum age for age group 1 in place types", "%i",
                        (void*)g_allParams.PlaceTypeAgeMin, g_allParams.PlaceTypeNum, 1, 0);
      GetInputParameter(PreParamFile_dat, AdminFile_dat, "Maximum age for age group 1 in place types", "%i",
                        (void*)g_allParams.PlaceTypeAgeMax, g_allParams.PlaceTypeNum, 1, 0);
      GetInputParameter(PreParamFile_dat, AdminFile_dat, "Proportion of age group 1 in place types", "%lf",
                        (void*)&(g_allParams.PlaceTypePropAgeGroup), g_allParams.PlaceTypeNum, 1, 0);
      if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Proportion of age group 2 in place types", "%lf",
                              (void*)&(g_allParams.PlaceTypePropAgeGroup2), g_allParams.PlaceTypeNum, 1, 0))
      {
         for (int i = 0; i < NUM_PLACE_TYPES; ++i)
         {
            g_allParams.PlaceTypePropAgeGroup2[i] = 0;
            g_allParams.PlaceTypeAgeMin2[i]       = 0;
            g_allParams.PlaceTypeAgeMax2[i]       = 1000;
         }
      }
      else
      {
         GetInputParameter(PreParamFile_dat, AdminFile_dat, "Minimum age for age group 2 in place types", "%i",
                           (void*)g_allParams.PlaceTypeAgeMin2, g_allParams.PlaceTypeNum, 1, 0);
         GetInputParameter(PreParamFile_dat, AdminFile_dat, "Maximum age for age group 2 in place types", "%i",
                           (void*)g_allParams.PlaceTypeAgeMax2, g_allParams.PlaceTypeNum, 1, 0);
      }
      if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Proportion of age group 3 in place types", "%lf",
                              (void*)&(g_allParams.PlaceTypePropAgeGroup3), g_allParams.PlaceTypeNum, 1, 0))
      {
         for (int i = 0; i < NUM_PLACE_TYPES; ++i)
         {
            g_allParams.PlaceTypePropAgeGroup3[i] = 0;
            g_allParams.PlaceTypeAgeMin3[i]       = 0;
            g_allParams.PlaceTypeAgeMax3[i]       = 1000;
         }
      }
      else
      {
         GetInputParameter(PreParamFile_dat, AdminFile_dat, "Minimum age for age group 3 in place types", "%i",
                           (void*)g_allParams.PlaceTypeAgeMin3, g_allParams.PlaceTypeNum, 1, 0);
         GetInputParameter(PreParamFile_dat, AdminFile_dat, "Maximum age for age group 3 in place types", "%i",
                           (void*)g_allParams.PlaceTypeAgeMax3, g_allParams.PlaceTypeNum, 1, 0);
      }
      if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Kernel shape params for place types", "%lf",
                              (void*)&(g_allParams.PlaceTypeKernelShape), g_allParams.PlaceTypeNum, 1, 0))
      {
         for (int i = 0; i < NUM_PLACE_TYPES; ++i)
         {
            g_allParams.PlaceTypeKernelShape[i] = g_allParams.MoveKernelShape;
            g_allParams.PlaceTypeKernelScale[i] = g_allParams.MoveKernelScale;
         }
      }
      else
      {
         GetInputParameter(PreParamFile_dat, AdminFile_dat, "Kernel scale params for place types", "%lf",
                           (void*)&(g_allParams.PlaceTypeKernelScale), g_allParams.PlaceTypeNum, 1, 0);
      }

      if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Kernel 3rd param for place types", "%lf",
                              (void*)&(g_allParams.PlaceTypeKernelP3), g_allParams.PlaceTypeNum, 1, 0))
      {
         for (int i = 0; i < NUM_PLACE_TYPES; ++i)
         {
            g_allParams.PlaceTypeKernelP3[i] = g_allParams.MoveKernelP3;
            g_allParams.PlaceTypeKernelP4[i] = g_allParams.MoveKernelP4;
         }
      }
      else
      {
         GetInputParameter(PreParamFile_dat, AdminFile_dat, "Kernel 4th param for place types", "%lf",
                           (void*)&(g_allParams.PlaceTypeKernelP4), g_allParams.PlaceTypeNum, 1, 0);
      }

      if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat,
                              "Number of closest places people pick from (0=all) for place types", "%i",
                              (void*)&(g_allParams.PlaceTypeNearestNeighb), g_allParams.PlaceTypeNum, 1, 0))
      {
         for (int i = 0; i < NUM_PLACE_TYPES; ++i)
         {
            g_allParams.PlaceTypeNearestNeighb[i] = 0;
         }
      }

      if (g_allParams.DoAdUnits)
      {
         if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                                 "Degree to which crossing administrative unit boundaries to go to places is inhibited",
                                 "%lf", (void*)&(g_allParams.InhibitInterAdunitPlaceAssignment),
                                 g_allParams.PlaceTypeNum, 1, 0))
         {
            for (int i = 0; i < NUM_PLACE_TYPES; ++i)
            {
               g_allParams.InhibitInterAdunitPlaceAssignment[i] = 0;
            }
         }
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Include air travel", "%i",
                              (void*)&(g_allParams.DoAirports), 1, 1, 0))
      {
         g_allParams.DoAirports = 0;
      }

      if (!g_allParams.DoAirports)
      {
         // Airports disabled => all places are not to do with airports, and we
         // have no hotels.
         g_allParams.PlaceTypeNoAirNum = g_allParams.PlaceTypeNum;
         g_allParams.HotelPlaceType    = g_allParams.PlaceTypeNum;
      }
      else
      {
         // When airports are activated we must have at least one airport place
         // // and a hotel type.
         GetInputParameter(PreParamFile_dat, AdminFile_dat, "Number of non-airport places", "%i",
                           (void*)&(g_allParams.PlaceTypeNoAirNum), 1, 1, 0);
         GetInputParameter(PreParamFile_dat, AdminFile_dat, "Hotel place type", "%i",
                           (void*)&(g_allParams.HotelPlaceType), 1, 1, 0);
         if (g_allParams.PlaceTypeNoAirNum >= g_allParams.PlaceTypeNum)
         {
            ERR_CRITICAL_FMT("[Number of non-airport places] parameter (%d) is greater than number of places (%d).\n",
                             g_allParams.PlaceTypeNoAirNum, g_allParams.PlaceTypeNum);
         }

         if (g_allParams.HotelPlaceType < g_allParams.PlaceTypeNoAirNum
             || g_allParams.HotelPlaceType >= g_allParams.PlaceTypeNum)
         {
            ERR_CRITICAL_FMT("[Hotel place type] parameter (%d) not in the range [%d, %d)\n",
                             g_allParams.HotelPlaceType, g_allParams.PlaceTypeNoAirNum, g_allParams.PlaceTypeNum);
         }

         if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                                 "Scaling factor for input file to convert to daily traffic", "%lf",
                                 (void*)&(g_allParams.AirportTrafficScale), 1, 1, 0))
         {
            g_allParams.AirportTrafficScale = 1.0;
         }

         if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Proportion of hotel attendees who are local", "%lf",
                                 (void*)&(g_allParams.HotelPropLocal), 1, 1, 0))
         {
            g_allParams.HotelPropLocal = 0;
         }

         if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Distribution of duration of air journeys", "%lf",
                                 (void*)&(g_allParams.JourneyDurationDistrib), MAX_TRAVEL_TIME, 1, 0))
         {
            g_allParams.JourneyDurationDistrib[0] = 1;
            for (int i = 0; i < MAX_TRAVEL_TIME; ++i)
            {
               g_allParams.JourneyDurationDistrib[i] = 0;
            }
         }
         if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Distribution of duration of local journeys", "%lf",
                                 (void*)&(g_allParams.LocalJourneyDurationDistrib), MAX_TRAVEL_TIME, 1, 0))
         {
            g_allParams.LocalJourneyDurationDistrib[0] = 1;
            for (int i = 0; i < MAX_TRAVEL_TIME; ++i)
            {
               g_allParams.LocalJourneyDurationDistrib[i] = 0;
            }
         }

         g_allParams.MeanJourneyTime      = 0;
         g_allParams.MeanLocalJourneyTime = 0;
         for (int i = 0; i < MAX_TRAVEL_TIME; ++i)
         {
            g_allParams.MeanJourneyTime += ((double)(i)) * g_allParams.JourneyDurationDistrib[i];
            g_allParams.MeanLocalJourneyTime += ((double)(i)) * g_allParams.LocalJourneyDurationDistrib[i];
         }

         fprintf(stderr, "Mean duration of local journeys = %lf days\n", g_allParams.MeanLocalJourneyTime);
         for (int i = 1; i < MAX_TRAVEL_TIME; ++i)
         {
            g_allParams.JourneyDurationDistrib[i] += g_allParams.JourneyDurationDistrib[i - 1];
            g_allParams.LocalJourneyDurationDistrib[i] += g_allParams.LocalJourneyDurationDistrib[i - 1];
         }

         for (int i = 0, j = 0; i <= 1024; ++i)
         {
            s = ((double)i) / 1024;
            while (g_allParams.JourneyDurationDistrib[j] < s)
            {
               ++j;
            }
            g_allParams.InvJourneyDurationDistrib[i] = j;
         }

         for (int i = 0, j = 0; i <= 1024; ++i)
         {
            s = ((double)i) / 1024;
            while (g_allParams.LocalJourneyDurationDistrib[j] < s)
            {
               ++j;
            }

            g_allParams.InvLocalJourneyDurationDistrib[i] = j;
         }
      }

      GetInputParameter(PreParamFile_dat, AdminFile_dat, "Mean size of place types", "%lf",
                        (void*)g_allParams.PlaceTypeMeanSize, g_allParams.PlaceTypeNum, 1, 0);
      GetInputParameter(PreParamFile_dat, AdminFile_dat, "Param 1 of place group size distribution", "%lf",
                        (void*)g_allParams.PlaceTypeGroupSizeParam1, g_allParams.PlaceTypeNum, 1, 0);

      if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Power of place size distribution", "%lf",
                              (void*)g_allParams.PlaceTypeSizePower, g_allParams.PlaceTypeNum, 1, 0))
      {
         for (int i = 0; i < NUM_PLACE_TYPES; ++i)
         {
            g_allParams.PlaceTypeSizePower[i] = 0;
         }
      }

      // added to enable lognormal distribution - ggilani 09/02/17
      if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Standard deviation of place size distribution", "%lf",
                              (void*)g_allParams.PlaceTypeSizeSD, g_allParams.PlaceTypeNum, 1, 0))
      {
         for (int i = 0; i < NUM_PLACE_TYPES; ++i)
         {
            g_allParams.PlaceTypeSizeSD[i] = 0;
         }
      }

      if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Offset of place size distribution", "%lf",
                              (void*)g_allParams.PlaceTypeSizeOffset, g_allParams.PlaceTypeNum, 1, 0))
      {
         for (int i = 0; i < NUM_PLACE_TYPES; ++i)
         {
            g_allParams.PlaceTypeSizeOffset[i] = 0;
         }
      }

      if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Maximum of place size distribution", "%lf",
                              (void*)g_allParams.PlaceTypeSizeMax, g_allParams.PlaceTypeNum, 1, 0))
      {
         for (int i = 0; i < NUM_PLACE_TYPES; ++i)
         {
            g_allParams.PlaceTypeSizeMax[i] = 1e20;
         }
      }

      if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Kernel type for place types", "%i",
                              (void*)g_allParams.PlaceTypeKernelType, g_allParams.PlaceTypeNum, 1, 0))
      {
         for (int i = 0; i < NUM_PLACE_TYPES; ++i)
         {
            g_allParams.PlaceTypeKernelType[i] = g_allParams.moveKernelType;
         }
      }

      GetInputParameter(PreParamFile_dat, AdminFile_dat, "Place overlap matrix", "%lf",
                        (void*)g_allParams.PlaceExclusivityMatrix, g_allParams.PlaceTypeNum * g_allParams.PlaceTypeNum,
                        1, 0); // changed from g_allParams.PlaceTypeNum,g_allParams.PlaceTypeNum,0);
      /* Note g_allParams.PlaceExclusivityMatrix not used at present - places assumed exclusive (each person belongs to
       * 0 or 1 place) */

      GetInputParameter(ParamFile_dat, PreParamFile_dat, "Proportion of between group place links", "%lf",
                        (void*)g_allParams.PlaceTypePropBetweenGroupLinks, g_allParams.PlaceTypeNum, 1, 0);
      GetInputParameter(ParamFile_dat, PreParamFile_dat, "Relative transmission rates for place types", "%lf",
                        (void*)g_allParams.PlaceTypeTrans, g_allParams.PlaceTypeNum, 1, 0);

      for (int i = 0; i < g_allParams.PlaceTypeNum; ++i)
      {
         g_allParams.PlaceTypeTrans[i] *= AgeSuscScale;
      }
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Daily seasonality coefficients", "%lf",
                           (void*)g_allParams.Seasonality, DAYS_PER_YEAR, 1, 0))
   {
      g_allParams.DoSeasonality = 0;
      for (int i = 0; i < DAYS_PER_YEAR; ++i)
      {
         g_allParams.Seasonality[i] = 1;
      }
   }
   else
   {
      g_allParams.DoSeasonality = 1;
      s                         = 0;
      for (int i = 0; i < DAYS_PER_YEAR; ++i)
      {
         s += g_allParams.Seasonality[i];
      }

      s += 1e-20;
      s /= DAYS_PER_YEAR;
      for (int i = 0; i < DAYS_PER_YEAR; ++i)
      {
         g_allParams.Seasonality[i] /= s;
      }
   }

   if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Number of seed locations", "%i",
                           (void*)&(g_allParams.NumSeedLocations), 1, 1, 0))
   {
      g_allParams.NumSeedLocations = 1;
   }

   if (g_allParams.NumSeedLocations > MAX_NUM_SEED_LOCATIONS)
   {
      fprintf(stderr, "Too many seed locations\n");
      g_allParams.NumSeedLocations = MAX_NUM_SEED_LOCATIONS;
   }

   GetInputParameter(PreParamFile_dat, AdminFile_dat, "Initial number of infecteds", "%i",
                     (void*)g_allParams.NumInitialInfections, g_allParams.NumSeedLocations, 1, 0);

   if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Location of initial infecteds", "%lf",
                           (void*)&(g_allParams.LocationInitialInfection[0][0]), g_allParams.NumSeedLocations * 2, 1,
                           0))
   {
      g_allParams.LocationInitialInfection[0][0] = 0.0;
      g_allParams.LocationInitialInfection[0][1] = 0.0;
   }

   if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Minimum population in microcell of initial infection",
                           "%i", (void*)&(g_allParams.MinPopDensForInitialInfection), 1, 1, 0))
   {
      g_allParams.MinPopDensForInitialInfection = 0;
   }

   if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Maximum population in microcell of initial infection",
                           "%i", (void*)&(g_allParams.MaxPopDensForInitialInfection), 1, 1, 0))
   {
      g_allParams.MaxPopDensForInitialInfection = 10000000;
   }

   if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Randomise initial infection location", "%i",
                           (void*)&(g_allParams.DoRandomInitialInfectionLoc), 1, 1, 0))
   {
      g_allParams.DoRandomInitialInfectionLoc = 1;
   }

   if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "All initial infections located in same microcell", "%i",
                           (void*)&(g_allParams.DoAllInitialInfectioninSameLoc), 1, 1, 0))
   {
      g_allParams.DoAllInitialInfectioninSameLoc = 0;
   }

   if (g_allParams.DoAdUnits)
   {
      if (!GetInputParameter2(
             PreParamFile_dat, AdminFile_dat, "Administrative unit to seed initial infection into", "%s",
             (g_allParams.NumSeedLocations > 1) ? ((void*)AdunitListNames) : ((void*)AdunitListNames[0]),
             g_allParams.NumSeedLocations, 1, 0))
      {
         for (int i = 0; i < g_allParams.NumSeedLocations; ++i)
         {
            g_allParams.InitialInfectionsAdminUnit[i] = 0;
         }
      }
      else
      {
         for (int i = 0; i < g_allParams.NumSeedLocations; ++i)
         {
            f = 0;
            if (g_allParams.NumAdunits > 0)
            {
               int j = 0;
               for (; (j < g_allParams.NumAdunits) && (!f); j++)
               {
                  f = (!strcmp(AdUnits[j].ad_name, AdunitListNames[i]));
               }

               if (f)
               {
                  k = AdUnits[j - 1].id;
               }
            }

            if (!f)
            {
               k = atoi(AdunitListNames[i]);
            }

            g_allParams.InitialInfectionsAdminUnit[i] = k;
            g_allParams.InitialInfectionsAdminUnitId[i] =
               g_allParams.AdunitLevel1Lookup[(k % g_allParams.AdunitLevel1Mask) / g_allParams.AdunitLevel1Divisor];
         }
      }

      if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Administrative unit seeding weights", "%lf",
                              (void*)&(g_allParams.InitialInfectionsAdminUnitWeight[0]), g_allParams.NumSeedLocations,
                              1, 0))
      {
         for (int i = 0; i < g_allParams.NumSeedLocations; i++)
         {
            g_allParams.InitialInfectionsAdminUnitWeight[i] = 1.0;
         }
      }

      s = 0;
      for (int i = 0; i < g_allParams.NumSeedLocations; ++i)
      {
         s += g_allParams.InitialInfectionsAdminUnitWeight[i];
      }

      for (int i = 0; i < g_allParams.NumSeedLocations; ++i)
      {
         g_allParams.InitialInfectionsAdminUnitWeight[i] /= s;
      }

      //	for (i = 0; i < g_allParams.NumSeedLocations; i++) fprintf(stderr, "## %i %s %i %i %lf\n",i,
      //AdUnits[g_allParams.InitialInfectionsAdminUnitId[i]].ad_name, g_allParams.InitialInfectionsAdminUnitId[i],
      //g_allParams.InitialInfectionsAdminUnit[i], g_allParams.InitialInfectionsAdminUnitWeight[i]);
   }
   else
   {
      for (int i = 0; i < g_allParams.NumSeedLocations; ++i)
      {
         g_allParams.InitialInfectionsAdminUnit[i] = 0;
      }
   }
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Initial rate of importation of infections", "%lf",
                           (void*)&(g_allParams.InfectionImportRate1), 1, 1, 0))
   {
      g_allParams.InfectionImportRate1 = 0;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Changed rate of importation of infections", "%lf",
                           (void*)&(g_allParams.InfectionImportRate2), 1, 1, 0))
   {
      g_allParams.InfectionImportRate2 = 0;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Time when infection rate changes", "%lf",
                           (void*)&(g_allParams.InfectionImportChangeTime), 1, 1, 0))
   {
      g_allParams.InfectionImportChangeTime = 1e10;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Imports via air travel", "%i",
                           (void*)&(g_allParams.DoImportsViaAirports), 1, 1, 0))
   {
      g_allParams.DoImportsViaAirports = 0;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Length of importation time profile provided", "%i",
                           (void*)&(g_allParams.DurImportTimeProfile), 1, 1, 0))
   {
      g_allParams.DurImportTimeProfile = 0;
   }

   if (g_allParams.DurImportTimeProfile > 0)
   {
      if (g_allParams.DurImportTimeProfile >= MAX_DUR_IMPORT_PROFILE)
      {
         ERR_CRITICAL("MAX_DUR_IMPORT_PROFILE too small\n");
      }

      GetInputParameter(ParamFile_dat, PreParamFile_dat, "Daily importation time profile", "%lf",
                        (void*)g_allParams.ImportInfectionTimeProfile, g_allParams.DurImportTimeProfile, 1, 0);
   }

   GetInputParameter(ParamFile_dat, PreParamFile_dat, "Reproduction number", "%lf", (void*)&(g_allParams.R0), 1, 1, 0);
   GetInputParameter(ParamFile_dat, PreParamFile_dat, "Infectious period", "%lf",
                     (void*)&(g_allParams.InfectiousPeriod), 1, 1, 0);

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "SD of individual variation in infectiousness", "%lf",
                           (void*)&(g_allParams.InfectiousnessSD), 1, 1, 0))
   {
      g_allParams.InfectiousnessSD = 0;
   }

   if (GetInputParameter2(ParamFile_dat, PreParamFile_dat, "k of individual variation in infectiousness", "%lf",
                          (void*)&s, 1, 1, 0))
   {
      g_allParams.InfectiousnessSD = 1.0 / sqrt(s);
   }

   if (g_allParams.InfectiousnessSD > 0)
   {
      g_allParams.InfectiousnessGamA = g_allParams.InfectiousnessGamR =
         1 / (g_allParams.InfectiousnessSD * g_allParams.InfectiousnessSD);
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Model time varying infectiousness", "%i",
                           (void*)&(g_allParams.DoInfectiousnessProfile), 1, 1, 0))
   {
      g_allParams.DoInfectiousnessProfile = 0;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Power of scaling of spatial R0 with density", "%lf",
                           (void*)&(g_allParams.R0DensityScalePower), 1, 1, 0))
   {
      g_allParams.R0DensityScalePower = 0;
   }

   if (g_allParams.DoInfectiousnessProfile)
   {
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Infectiousness profile", "%lf",
                              (void*)g_allParams.infectious_prof, INFPROF_RES, 1, 0))
      {
         for (int i = 0; i < INFPROF_RES; ++i)
         {
            g_allParams.infectious_prof[i] = 1;
         }
      }

      k = static_cast<int>(ceil(g_allParams.InfectiousPeriod / g_allParams.TimeStep));
      if (k >= MAX_INFECTIOUS_STEPS)
      {
         ERR_CRITICAL("MAX_INFECTIOUS_STEPS not big enough\n");
      }

      s                                        = 0;
      g_allParams.infectious_prof[INFPROF_RES] = 0;
      for (int i = 0; i < MAX_INFECTIOUS_STEPS; ++i)
      {
         g_allParams.infectiousness[i] = 0;
      }

      for (int i = 0; i < k; ++i)
      {
         t = (((double)i) * g_allParams.TimeStep / g_allParams.InfectiousPeriod * INFPROF_RES);
         int j = (int)t;
         t -= (double)j;
         if (j < INFPROF_RES)
         {
            s += (g_allParams.infectiousness[i] =
                     g_allParams.infectious_prof[j] * (1 - t) + g_allParams.infectious_prof[j + 1] * t);
         }
         else
         {
            s += (g_allParams.infectiousness[i] = g_allParams.infectious_prof[INFPROF_RES]);
         }
      }

      s /= ((double)k);
      for (int i = 0; i <= k; ++i)
      {
         g_allParams.infectiousness[i] /= s;
      }

      for (int i = 0; i <= CDF_RES; ++i)
      {
         g_allParams.infectious_icdf[i] = exp(-1.0);
      }
   }
   else
   {
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Infectious period inverse CDF", "%lf",
                              (void*)g_allParams.infectious_icdf, CDF_RES + 1, 1, 0))
      {
         g_allParams.infectious_icdf[CDF_RES] = 100;
         for (int i = 0; i < CDF_RES; ++i)
         {
            g_allParams.infectious_icdf[i] = -log(1 - ((double)i) / CDF_RES);
         }
      }
      k = (int)ceil(g_allParams.InfectiousPeriod * g_allParams.infectious_icdf[CDF_RES] / g_allParams.TimeStep);
      if (k >= MAX_INFECTIOUS_STEPS)
      {
         ERR_CRITICAL("MAX_INFECTIOUS_STEPS not big enough\n");
      }

      for (int i = 0; i < k; ++i)
      {
         g_allParams.infectiousness[i] = 1.0;
      }

      g_allParams.infectiousness[k] = 0;
      for (int i = 0; i <= CDF_RES; ++i)
      {
         g_allParams.infectious_icdf[i] = exp(-g_allParams.infectious_icdf[i]);
      }
   }
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Include latent period", "%i",
                           (void*)&(g_allParams.DoLatent), 1, 1, 0))
   {
      g_allParams.DoLatent = 0;
   }

   if (g_allParams.DoLatent)
   {
      GetInputParameter(ParamFile_dat, PreParamFile_dat, "Latent period", "%lf", (void*)&(g_allParams.LatentPeriod), 1,
                        1, 0);
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Latent period inverse CDF", "%lf",
                              (void*)g_allParams.latent_icdf, CDF_RES + 1, 1, 0))
      {
         g_allParams.latent_icdf[CDF_RES] = 1e10;
         for (int i = 0; i < CDF_RES; ++i)
         {
            g_allParams.latent_icdf[i] = -log(1 - ((double)i) / CDF_RES);
         }
      }

      for (int i = 0; i <= CDF_RES; ++i)
      {
         g_allParams.latent_icdf[i] = exp(-g_allParams.latent_icdf[i]);
      }
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Include symptoms", "%i", (void*)&(g_allParams.DoSymptoms),
                           1, 1, 0))
   {
      g_allParams.DoSymptoms = 0;
   }

   if (!g_allParams.DoSymptoms)
   {
      for (int i = 0; i < NUM_AGE_GROUPS; ++i)
      {
         g_allParams.ProportionSymptomatic[i] = 0;
      }

      g_allParams.FalsePositiveRate   = 0;
      g_allParams.SymptInfectiousness = 1.0;
      g_allParams.LatentToSymptDelay  = 0;
   }
   else
   {
      if (g_allParams.DoAge)
      {
         GetInputParameter(ParamFile_dat, PreParamFile_dat, "Proportion symptomatic by age group", "%lf",
                           (void*)g_allParams.ProportionSymptomatic, NUM_AGE_GROUPS, 1, 0);
      }
      else
      {
         GetInputParameter(ParamFile_dat, PreParamFile_dat, "Proportion symptomatic", "%lf",
                           (void*)g_allParams.ProportionSymptomatic, 1, 1, 0);
         for (int i = 1; i < NUM_AGE_GROUPS; ++i)
         {
            g_allParams.ProportionSymptomatic[i] = g_allParams.ProportionSymptomatic[0];
         }
      }

      GetInputParameter(ParamFile_dat, PreParamFile_dat, "Delay from end of latent period to start of symptoms", "%lf",
                        (void*)&(g_allParams.LatentToSymptDelay), 1, 1, 0);
      GetInputParameter(ParamFile_dat, PreParamFile_dat, "Relative rate of random contacts if symptomatic", "%lf",
                        (void*)&(g_allParams.SymptSpatialContactRate), 1, 1, 0);
      GetInputParameter(ParamFile_dat, PreParamFile_dat, "Symptomatic infectiousness relative to asymptomatic", "%lf",
                        (void*)&(g_allParams.SymptInfectiousness), 1, 1, 0);

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Model symptomatic withdrawal to home as true absenteeism", "%i",
                              (void*)&g_allParams.DoRealSymptWithdrawal, 1, 1, 0))
      {
         g_allParams.DoRealSymptWithdrawal = 0;
      }

      if (g_allParams.DoPlaces)
      {
         GetInputParameter(ParamFile_dat, PreParamFile_dat, "Relative level of place attendance if symptomatic", "%lf",
                           (void*)g_allParams.SymptPlaceTypeContactRate, g_allParams.PlaceTypeNum, 1, 0);
         if (g_allParams.DoRealSymptWithdrawal)
         {
            for (int j = 0; j < NUM_PLACE_TYPES; ++j)
            {
               g_allParams.SymptPlaceTypeWithdrawalProp[j] = 1.0 - g_allParams.SymptPlaceTypeContactRate[j];
               g_allParams.SymptPlaceTypeContactRate[j]    = 1.0;
            }
         }
         else
         {
            for (int j = 0; j < NUM_PLACE_TYPES; ++j)
            {
               g_allParams.SymptPlaceTypeWithdrawalProp[j] = 0.0;
            }
         }
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Maximum age of child at home for whom one adult also stays at home", "%i",
                              (void*)&g_allParams.CaseAbsentChildAgeCutoff, 1, 1, 0))
      {
         g_allParams.CaseAbsentChildAgeCutoff = 0;
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Proportion of children at home for whom one adult also stays at home", "%lf",
                              (void*)&g_allParams.CaseAbsentChildPropAdultCarers, 1, 1, 0))
      {
         g_allParams.CaseAbsentChildPropAdultCarers = 0;
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Place close round household", "%i",
                              (void*)&g_allParams.PlaceCloseRoundHousehold, 1, 1, 0))
      {
         g_allParams.PlaceCloseRoundHousehold = 1;
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Absenteeism place closure", "%i",
                              (void*)&g_allParams.AbsenteeismPlaceClosure, 1, 1, 0))
      {
         g_allParams.AbsenteeismPlaceClosure = 0;
      }

      if (g_allParams.AbsenteeismPlaceClosure)
      {
         g_allParams.CaseAbsenteeismDelay = 0; // Set to zero for tracking absenteeism
         if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Max absent time", "%i",
                                 (void*)&g_allParams.MaxAbsentTime, 1, 1, 0))
         {
            g_allParams.MaxAbsentTime = MAX_ABSENT_TIME;
         }

         if (g_allParams.MaxAbsentTime > MAX_ABSENT_TIME || g_allParams.MaxAbsentTime < 0)
         {
            ERR_CRITICAL_FMT("[Max absent time] out of range (%d), should be in range [0, %d]",
                             g_allParams.MaxAbsentTime, MAX_ABSENT_TIME);
         }
      }
      else
      {
         if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                                 "Delay in starting place absenteeism for cases who withdraw", "%lf",
                                 (void*)&g_allParams.CaseAbsenteeismDelay, 1, 1, 0))
         {
            g_allParams.CaseAbsenteeismDelay = 0;
         }

         g_allParams.MaxAbsentTime = 0; // Not used when !g_allParams.AbsenteeismPlaceClosure
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Duration of place absenteeism for cases who withdraw",
                              "%lf", (void*)&g_allParams.CaseAbsenteeismDuration, 1, 1, 0))
      {
         g_allParams.CaseAbsenteeismDuration = 7;
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "False positive rate", "%lf",
                              (void*)&(g_allParams.FalsePositiveRate), 1, 1, 0))
      {
         g_allParams.FalsePositiveRate = 0.0;
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "False positive per capita incidence", "%lf",
                              (void*)&(g_allParams.FalsePositivePerCapitaIncidence), 1, 1, 0))
      {
         g_allParams.FalsePositivePerCapitaIncidence = 0.0;
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "False positive relative incidence by age", "%lf",
                              (void*)g_allParams.FalsePositiveAgeRate, NUM_AGE_GROUPS, 1, 0))
      {
         for (int j = 0; j < NUM_AGE_GROUPS; ++j)
            g_allParams.FalsePositiveAgeRate[j] = 1.0;
      }
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Do Severity Analysis", "%i",
                           (void*)&(g_allParams.DoSeverity), 1, 1, 0))
   {
      g_allParams.DoSeverity = 0;
   }

   if (g_allParams.DoSeverity == 1)
   {
      //// Means for icdf's.
      GetInputParameter(ParamFile_dat, PreParamFile_dat, "Mean_MildToRecovery", "%lf",
                        (void*)&(g_allParams.Mean_MildToRecovery), 1, 1, 0);
      GetInputParameter(ParamFile_dat, PreParamFile_dat, "Mean_ILIToRecovery", "%lf",
                        (void*)&(g_allParams.Mean_ILIToRecovery), 1, 1, 0);
      GetInputParameter(ParamFile_dat, PreParamFile_dat, "Mean_SARIToRecovery", "%lf",
                        (void*)&(g_allParams.Mean_SARIToRecovery), 1, 1, 0);
      GetInputParameter(ParamFile_dat, PreParamFile_dat, "Mean_CriticalToCritRecov", "%lf",
                        (void*)&(g_allParams.Mean_CriticalToCritRecov), 1, 1, 0);
      GetInputParameter(ParamFile_dat, PreParamFile_dat, "Mean_CritRecovToRecov", "%lf",
                        (void*)&(g_allParams.Mean_CritRecovToRecov), 1, 1, 0);
      GetInputParameter(ParamFile_dat, PreParamFile_dat, "Mean_ILIToSARI", "%lf", (void*)&(g_allParams.Mean_ILIToSARI),
                        1, 1, 0);

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Mean_ILIToDeath", "%lf",
                              (void*)&(g_allParams.Mean_ILIToDeath), 1, 1, 0))
      {
         g_allParams.Mean_ILIToDeath = 7.0;
      }

      GetInputParameter(ParamFile_dat, PreParamFile_dat, "Mean_SARIToCritical", "%lf",
                        (void*)&(g_allParams.Mean_SARIToCritical), 1, 1, 0);
      GetInputParameter(ParamFile_dat, PreParamFile_dat, "Mean_SARIToDeath", "%lf",
                        (void*)&(g_allParams.Mean_SARIToDeath), 1, 1, 0);
      GetInputParameter(ParamFile_dat, PreParamFile_dat, "Mean_CriticalToDeath", "%lf",
                        (void*)&(g_allParams.Mean_CriticalToDeath), 1, 1, 0);

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "MeanTimeToTest", "%lf",
                              (void*)&(g_allParams.Mean_TimeToTest), 1, 1, 0))
      {
         g_allParams.Mean_TimeToTest = 0.0;
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "MeanTimeToTestOffset", "%lf",
                              (void*)&(g_allParams.Mean_TimeToTestOffset), 1, 1, 0))
      {
         g_allParams.Mean_TimeToTestOffset = 1.0;
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "MeanTimeToTestCriticalOffset", "%lf",
                              (void*)&(g_allParams.Mean_TimeToTestCriticalOffset), 1, 1, 0))
      {
         g_allParams.Mean_TimeToTestCriticalOffset = 1.0;
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "MeanTimeToTestCritRecovOffset", "%lf",
                              (void*)&(g_allParams.Mean_TimeToTestCritRecovOffset), 1, 1, 0))
      {
         g_allParams.Mean_TimeToTestCritRecovOffset = 1.0;
      }

      //// Get ICDFs
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "MildToRecovery_icdf", "%lf",
                              (void*)g_allParams.MildToRecovery_icdf, CDF_RES + 1, 1, 0))
      {
         g_allParams.MildToRecovery_icdf[CDF_RES] = 100;
         for (int i = 0; i < CDF_RES; ++i)
         {
            g_allParams.MildToRecovery_icdf[i] = -log(1 - ((double)i) / CDF_RES);
         }
      }

      for (int i = 0; i <= CDF_RES; ++i)
      {
         g_allParams.MildToRecovery_icdf[i] = exp(-g_allParams.MildToRecovery_icdf[i]);
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "ILIToRecovery_icdf", "%lf",
                              (void*)g_allParams.ILIToRecovery_icdf, CDF_RES + 1, 1, 0))
      {
         g_allParams.ILIToRecovery_icdf[CDF_RES] = 100;
         for (int i = 0; i < CDF_RES; ++i)
         {
            g_allParams.ILIToRecovery_icdf[i] = -log(1 - ((double)i) / CDF_RES);
         }
      }

      for (int i = 0; i <= CDF_RES; ++i)
      {
         g_allParams.ILIToRecovery_icdf[i] = exp(-g_allParams.ILIToRecovery_icdf[i]);
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "ILIToDeath_icdf", "%lf",
                              (void*)g_allParams.ILIToDeath_icdf, CDF_RES + 1, 1, 0))
      {
         g_allParams.ILIToDeath_icdf[CDF_RES] = 100;
         for (int i = 0; i < CDF_RES; ++i)
         {
            g_allParams.ILIToDeath_icdf[i] = -log(1 - ((double)i) / CDF_RES);
         }
      }

      for (int i = 0; i <= CDF_RES; ++i)
      {
         g_allParams.ILIToDeath_icdf[i] = exp(-g_allParams.ILIToDeath_icdf[i]);
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "SARIToRecovery_icdf", "%lf",
                              (void*)g_allParams.SARIToRecovery_icdf, CDF_RES + 1, 1, 0))
      {
         g_allParams.SARIToRecovery_icdf[CDF_RES] = 100;
         for (int i = 0; i < CDF_RES; ++i)
         {
            g_allParams.SARIToRecovery_icdf[i] = -log(1 - ((double)i) / CDF_RES);
         }
      }

      for (int i = 0; i <= CDF_RES; ++i)
      {
         g_allParams.SARIToRecovery_icdf[i] = exp(-g_allParams.SARIToRecovery_icdf[i]);
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "CriticalToCritRecov_icdf", "%lf",
                              (void*)g_allParams.CriticalToCritRecov_icdf, CDF_RES + 1, 1, 0))
      {
         g_allParams.CriticalToCritRecov_icdf[CDF_RES] = 100;
         for (int i = 0; i < CDF_RES; ++i)
         {
            g_allParams.CriticalToCritRecov_icdf[i] = -log(1 - ((double)i) / CDF_RES);
         }
      }

      for (int i = 0; i <= CDF_RES; ++i)
      {
         g_allParams.CriticalToCritRecov_icdf[i] = exp(-g_allParams.CriticalToCritRecov_icdf[i]);
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "CritRecovToRecov_icdf", "%lf",
                              (void*)g_allParams.CritRecovToRecov_icdf, CDF_RES + 1, 1, 0))
      {
         g_allParams.CritRecovToRecov_icdf[CDF_RES] = 100;
         for (int i = 0; i < CDF_RES; ++i)
         {
            g_allParams.CritRecovToRecov_icdf[i] = -log(1 - ((double)i) / CDF_RES);
         }
      }

      for (int i = 0; i <= CDF_RES; ++i)
      {
         g_allParams.CritRecovToRecov_icdf[i] = exp(-g_allParams.CritRecovToRecov_icdf[i]);
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "ILIToSARI_icdf", "%lf",
                              (void*)g_allParams.ILIToSARI_icdf, CDF_RES + 1, 1, 0))
      {
         g_allParams.ILIToSARI_icdf[CDF_RES] = 100;
         for (int i = 0; i < CDF_RES; ++i)
         {
            g_allParams.ILIToSARI_icdf[i] = -log(1 - ((double)i) / CDF_RES);
         }
      }

      for (int i = 0; i <= CDF_RES; ++i)
      {
         g_allParams.ILIToSARI_icdf[i] = exp(-g_allParams.ILIToSARI_icdf[i]);
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "SARIToCritical_icdf", "%lf",
                              (void*)g_allParams.SARIToCritical_icdf, CDF_RES + 1, 1, 0))
      {
         g_allParams.SARIToCritical_icdf[CDF_RES] = 100;
         for (int i = 0; i < CDF_RES; ++i)
         {
            g_allParams.SARIToCritical_icdf[i] = -log(1 - ((double)i) / CDF_RES);
         }
      }
      for (int i = 0; i <= CDF_RES; ++i)
      {
         g_allParams.SARIToCritical_icdf[i] = exp(-g_allParams.SARIToCritical_icdf[i]);
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "SARIToDeath_icdf", "%lf",
                              (void*)g_allParams.SARIToDeath_icdf, CDF_RES + 1, 1, 0))
      {
         g_allParams.SARIToDeath_icdf[CDF_RES] = 100;
         for (int i = 0; i < CDF_RES; ++i)
         {
            g_allParams.SARIToDeath_icdf[i] = -log(1 - ((double)i) / CDF_RES);
         }
      }

      for (int i = 0; i <= CDF_RES; ++i)
      {
         g_allParams.SARIToDeath_icdf[i] = exp(-g_allParams.SARIToDeath_icdf[i]);
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "CriticalToDeath_icdf", "%lf",
                              (void*)g_allParams.CriticalToDeath_icdf, CDF_RES + 1, 1, 0))
      {
         g_allParams.CriticalToDeath_icdf[CDF_RES] = 100;
         for (int i = 0; i < CDF_RES; ++i)
         {
            g_allParams.CriticalToDeath_icdf[i] = -log(1 - ((double)i) / CDF_RES);
         }
      }

      for (int i = 0; i <= CDF_RES; ++i)
      {
         g_allParams.CriticalToDeath_icdf[i] = exp(-g_allParams.CriticalToDeath_icdf[i]);
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Prop_Mild_ByAge", "%lf",
                              (void*)g_allParams.Prop_Mild_ByAge, NUM_AGE_GROUPS, 1, 0))
      {
         for (int i = 0; i < NUM_AGE_GROUPS; ++i)
         {
            g_allParams.Prop_Mild_ByAge[i] = 0.5;
         }
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Prop_ILI_ByAge", "%lf",
                              (void*)g_allParams.Prop_ILI_ByAge, NUM_AGE_GROUPS, 1, 0))
      {
         for (int i = 0; i < NUM_AGE_GROUPS; ++i)
         {
            g_allParams.Prop_ILI_ByAge[i] = 0.3;
         }
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Prop_SARI_ByAge", "%lf",
                              (void*)g_allParams.Prop_SARI_ByAge, NUM_AGE_GROUPS, 1, 0))
      {
         for (int i = 0; i < NUM_AGE_GROUPS; ++i)
         {
            g_allParams.Prop_SARI_ByAge[i] = 0.15;
         }
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Prop_Critical_ByAge", "%lf",
                              (void*)g_allParams.Prop_Critical_ByAge, NUM_AGE_GROUPS, 1, 0))
      {
         for (int i = 0; i < NUM_AGE_GROUPS; ++i)
         {
            g_allParams.Prop_Critical_ByAge[i] = 0.05;
         }
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "CFR_SARI_ByAge", "%lf",
                              (void*)g_allParams.CFR_SARI_ByAge, NUM_AGE_GROUPS, 1, 0))
      {
         for (int i = 0; i < NUM_AGE_GROUPS; ++i)
         {
            g_allParams.CFR_SARI_ByAge[i] = 0.50;
         }
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "CFR_Critical_ByAge", "%lf",
                              (void*)g_allParams.CFR_Critical_ByAge, NUM_AGE_GROUPS, 1, 0))
      {
         for (int i = 0; i < NUM_AGE_GROUPS; ++i)
         {
            g_allParams.CFR_Critical_ByAge[i] = 0.50;
         }
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "CFR_ILI_ByAge", "%lf", (void*)g_allParams.CFR_ILI_ByAge,
                              NUM_AGE_GROUPS, 1, 0))
      {
         for (int i = 0; i < NUM_AGE_GROUPS; ++i)
         {
            g_allParams.CFR_ILI_ByAge[i] = 0.00;
         }
      }
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Bounding box for bitmap", "%lf",
                           (void*)&(g_allParams.BoundingBox[0]), 4, 1, 0))
   {
      g_allParams.BoundingBox[0] = g_allParams.BoundingBox[1] = 0.0;
      g_allParams.BoundingBox[2] = g_allParams.BoundingBox[3] = 1.0;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Spatial domain for simulation", "%lf",
                           (void*)&(g_allParams.SpatialBoundingBox[0]), 4, 1, 0))
   {
      g_allParams.SpatialBoundingBox[0] = g_allParams.SpatialBoundingBox[1] = 0.0;
      g_allParams.SpatialBoundingBox[2] = g_allParams.SpatialBoundingBox[3] = 1.0;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Grid size", "%lf", (void*)&(g_allParams.cwidth), 1, 1, 0))
   {
      g_allParams.cwidth = 1.0 / 120.0;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Use long/lat coord system", "%i",
                           (void*)&(g_allParams.DoUTM_coords), 1, 1, 0))
   {
      g_allParams.DoUTM_coords = 1;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Bitmap scale", "%lf", (void*)&(g_allParams.BitmapScale), 1,
                           1, 0))
   {
      g_allParams.BitmapScale = 1.0;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Bitmap y:x aspect scaling", "%lf",
                           (void*)&(g_allParams.BitmapAspectScale), 1, 1, 0))
   {
      g_allParams.BitmapAspectScale = 1.0;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Bitmap movie frame interval", "%i",
                           (void*)&(g_allParams.BitmapMovieFrame), 1, 1, 0))
   {
      g_allParams.BitmapMovieFrame = 250;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Output bitmap", "%i", (void*)&(g_allParams.OutputBitmap),
                           1, 1, 0))
   {
      g_allParams.OutputBitmap = 0;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Output bitmap detected", "%i",
                           (void*)&(g_allParams.OutputBitmapDetected), 1, 1, 0))
   {
      g_allParams.OutputBitmapDetected = 0;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Output immunity on bitmap", "%i",
                           (void*)&(g_allParams.DoImmuneBitmap), 1, 1, 0))
   {
      g_allParams.DoImmuneBitmap = 0;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Output infection tree", "%i",
                           (void*)&(g_allParams.DoInfectionTree), 1, 1, 0))
   {
      g_allParams.DoInfectionTree = 0;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Do one generation", "%i", (void*)&(g_allParams.DoOneGen),
                           1, 1, 0))
   {
      g_allParams.DoOneGen = 0;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Output every realisation", "%i",
                           (void*)&(g_allParams.OutputEveryRealisation), 1, 1, 0))
   {
      g_allParams.OutputEveryRealisation = 0;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Maximum number to sample for correlations", "%i",
                           (void*)&(g_allParams.MaxCorrSample), 1, 1, 0))
   {
      g_allParams.MaxCorrSample = 1000000000;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Assume SI model", "%i", (void*)&(g_allParams.DoSI), 1, 1,
                           0))
   {
      g_allParams.DoSI = 0;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Assume periodic boundary conditions", "%i",
                           (void*)&(g_allParams.DoPeriodicBoundaries), 1, 1, 0))
   {
      g_allParams.DoPeriodicBoundaries = 0;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Only output non-extinct realisations", "%i",
                           (void*)&(g_allParams.OutputOnlyNonExtinct), 1, 1, 0))
   {
      g_allParams.OutputOnlyNonExtinct = 0;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Use cases per thousand threshold for area controls", "%i",
                           (void*)&(g_allParams.DoPerCapitaTriggers), 1, 1, 0))
   {
      g_allParams.DoPerCapitaTriggers = 0;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Use global triggers for interventions", "%i",
                           (void*)&(g_allParams.DoGlobalTriggers), 1, 1, 0))
   {
      g_allParams.DoGlobalTriggers = 0;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Use admin unit triggers for interventions", "%i",
                           (void*)&(g_allParams.DoAdminTriggers), 1, 1, 0))
   {
      g_allParams.DoAdminTriggers = 0;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Use ICU case triggers for interventions", "%i",
                           (void*)&(g_allParams.DoICUTriggers), 1, 1, 0))
   {
      g_allParams.DoICUTriggers = 0;
   }

   if (g_allParams.DoGlobalTriggers)
   {
      g_allParams.DoAdminTriggers = 0;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Divisor for per-capita area threshold (default 1000)",
                           "%i", (void*)&(g_allParams.IncThreshPop), 1, 1, 0))
   {
      g_allParams.IncThreshPop = 1000;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Divisor for per-capita global threshold (default 1000)",
                           "%i", (void*)&(g_allParams.GlobalIncThreshPop), 1, 1, 0))
   {
      g_allParams.GlobalIncThreshPop = 1000;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                           "Number of sampling intervals over which cumulative incidence measured for global trigger",
                           "%i", (void*)&(g_allParams.TriggersSamplingInterval), 1, 1, 0))
   {
      g_allParams.TriggersSamplingInterval = 10000000;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Proportion of cases detected for treatment", "%lf",
                           (void*)&(g_allParams.PostAlertControlPropCasesId), 1, 1, 0))
   {
      g_allParams.PostAlertControlPropCasesId = 1;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Proportion of cases detected before outbreak alert", "%lf",
                           (void*)&(g_allParams.PreAlertControlPropCasesId), 1, 1, 0))
   {
      g_allParams.PreAlertControlPropCasesId = 1.0;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Trigger alert on deaths", "%i",
                           (void*)&(g_allParams.PreControlClusterIdUseDeaths), 1, 1, 0))
   {
      g_allParams.PreControlClusterIdUseDeaths = 0;
   }

   if (g_allParams.PreControlClusterIdUseDeaths)
   {
      if (g_allParams.PreControlClusterIdCaseThreshold == 0)
      {
         if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Number of deaths accummulated before alert", "%i",
                                 (void*)&(g_allParams.PreControlClusterIdCaseThreshold), 1, 1, 0))
         {
            g_allParams.PreControlClusterIdCaseThreshold = 0;
         }
      }
   }
   else if (g_allParams.PreControlClusterIdCaseThreshold == 0)
   {
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Number of detected cases needed before outbreak alert triggered", "%i",
                              (void*)&(g_allParams.PreControlClusterIdCaseThreshold), 1, 1, 0))
      {
         g_allParams.PreControlClusterIdCaseThreshold = 0;
      }
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Alert trigger starts after interventions", "%i",
                           (void*)&(g_allParams.DoAlertTriggerAfterInterv), 1, 1, 0))
   {
      g_allParams.DoAlertTriggerAfterInterv = 0;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Day of year trigger is reached", "%lf",
                           (void*)&(g_allParams.PreControlClusterIdCalTime), 1, 1, 0))
   {
      g_allParams.PreControlClusterIdCalTime = -1;
   }

   if (g_allParams.DoAlertTriggerAfterInterv)
   {
      GetInputParameter(ParamFile_dat, PreParamFile_dat, "Day of year interventions start", "%lf",
                        (void*)&(g_allParams.PreIntervIdCalTime), 1, 1, 0);
      if (g_allParams.PreControlClusterIdCalTime <= g_allParams.PreIntervIdCalTime)
      {
         g_allParams.DoAlertTriggerAfterInterv = 0;
      }
      else
      {
         g_allParams.AlertTriggerAfterIntervThreshold = g_allParams.PreControlClusterIdCaseThreshold;
         g_allParams.PreControlClusterIdCaseThreshold = 1000;
      }
   }
   else
   {
      g_allParams.PreIntervIdCalTime = g_allParams.PreControlClusterIdCalTime;
   }

   g_allParams.StopCalibration = g_allParams.ModelCalibIteration = 0;
   g_allParams.SeedingScaling                                    = 1.0;
   g_allParams.PreControlClusterIdTime                           = 0;
   // if (g_allParams.DoAlertTriggerAfterInterv) g_allParams.ResetSeeds =g_allParams.KeepSameSeeds = 1;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Number of days to accummulate cases/deaths before alert",
                           "%i", (void*)&(g_allParams.PreControlClusterIdDuration), 1, 1, 0))
   {
      g_allParams.PreControlClusterIdDuration = 1000;
   }

   g_allParams.PreControlClusterIdHolOffset = 0;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Only use confirmed cases to trigger alert", "%i",
                           (void*)&(g_allParams.DoEarlyCaseDiagnosis), 1, 1, 0))
   {
      g_allParams.DoEarlyCaseDiagnosis = 0;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Only treat mixing groups within places", "%i",
                           (void*)&(g_allParams.DoPlaceGroupTreat), 1, 1, 0))
   {
      g_allParams.DoPlaceGroupTreat = 0;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Treatment trigger incidence per cell", "%lf",
                           (void*)&(g_allParams.TreatCellIncThresh), 1, 1, 0))
   {
      g_allParams.TreatCellIncThresh = 1000000000;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Case isolation trigger incidence per cell", "%lf",
                           (void*)&(g_allParams.CaseIsolation_CellIncThresh), 1, 1, 0))
   {
      g_allParams.CaseIsolation_CellIncThresh =
         g_allParams.TreatCellIncThresh; //// changed default to be g_allParams.TreatCellIncThresh
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Household quarantine trigger incidence per cell", "%lf",
                           (void*)&(g_allParams.HHQuar_CellIncThresh), 1, 1, 0))
   {
      g_allParams.HHQuar_CellIncThresh =
         g_allParams.TreatCellIncThresh; //// changed default to be g_allParams.TreatCellIncThresh
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Relative susceptibility of treated individual", "%lf",
                           (void*)&(g_allParams.TreatSuscDrop), 1, 1, 0))
   {
      g_allParams.TreatSuscDrop = 1;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Relative infectiousness of treated individual", "%lf",
                           (void*)&(g_allParams.TreatInfDrop), 1, 1, 0))
   {
      g_allParams.TreatInfDrop = 1;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                           "Proportion of symptomatic cases resulting in death prevented by treatment", "%lf",
                           (void*)&(g_allParams.TreatDeathDrop), 1, 1, 0))
   {
      g_allParams.TreatDeathDrop = 0;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Proportion of symptomatic cases prevented by treatment",
                           "%lf", (void*)&(g_allParams.TreatSympDrop), 1, 1, 0))
   {
      g_allParams.TreatSympDrop = 0;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Delay to treat cell", "%lf",
                           (void*)&(g_allParams.TreatDelayMean), 1, 1, 0))
   {
      g_allParams.TreatDelayMean = 0;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Duration of course of treatment", "%lf",
                           (void*)&(g_allParams.TreatCaseCourseLength), 1, 1, 0))
   {
      g_allParams.TreatCaseCourseLength = 5;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Duration of course of prophylaxis", "%lf",
                           (void*)&(g_allParams.TreatProphCourseLength), 1, 1, 0))
   {
      g_allParams.TreatProphCourseLength = 10;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Proportion of detected cases treated", "%lf",
                           (void*)&(g_allParams.TreatPropCases), 1, 1, 0))
   {
      g_allParams.TreatPropCases = 1;
   }

   if (g_allParams.DoHouseholds)
   {
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Proportion of households of cases treated", "%lf",
                              (void*)&(g_allParams.TreatPropCaseHouseholds), 1, 1, 0))
      {
         g_allParams.TreatPropCaseHouseholds = 0;
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Duration of household prophylaxis policy", "%lf",
                              (void*)&(g_allParams.TreatHouseholdsDuration), 1, 1, 0))
      {
         g_allParams.TreatHouseholdsDuration = USHRT_MAX / g_allParams.TimeStepsPerDay;
      }
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Proportion treated", "%lf",
                           (void*)&(g_allParams.TreatPropRadial), 1, 1, 0))
      g_allParams.TreatPropRadial = 1.0;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Proportion treated in radial prophylaxis", "%lf",
                           (void*)&(g_allParams.TreatPropRadial), 1, 1, 0))
      g_allParams.TreatPropRadial = 1.0;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Treatment radius", "%lf",
                           (void*)&(g_allParams.TreatRadius), 1, 1, 0))
      g_allParams.TreatRadius = 0;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Duration of place/geographic prophylaxis policy", "%lf",
                           (void*)&(g_allParams.TreatPlaceGeogDuration), 1, 1, 0))
      g_allParams.TreatPlaceGeogDuration = USHRT_MAX / g_allParams.TimeStepsPerDay;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Treatment start time", "%lf",
                           (void*)&(g_allParams.TreatTimeStartBase), 1, 1, 0))
      g_allParams.TreatTimeStartBase = USHRT_MAX / g_allParams.TimeStepsPerDay;
   if (g_allParams.DoPlaces)
   {
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Proportion of places treated after case detected",
                              "%lf", (void*)g_allParams.TreatPlaceProbCaseId, g_allParams.PlaceTypeNum, 1, 0))
         for (int i = 0; i < NUM_PLACE_TYPES; i++)
            g_allParams.TreatPlaceProbCaseId[i] = 0;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Proportion of people treated in targeted places", "%lf",
                              (void*)g_allParams.TreatPlaceTotalProp, g_allParams.PlaceTypeNum, 1, 0))
         for (int i = 0; i < NUM_PLACE_TYPES; i++)
            g_allParams.TreatPlaceTotalProp[i] = 0;
   }
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Maximum number of doses available", "%lf",
                           (void*)&(g_allParams.TreatMaxCoursesBase), 1, 1, 0))
      g_allParams.TreatMaxCoursesBase = 1e20;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Start time of additional treatment production", "%lf",
                           (void*)&(g_allParams.TreatNewCoursesStartTime), 1, 1, 0))
      g_allParams.TreatNewCoursesStartTime = USHRT_MAX / g_allParams.TimeStepsPerDay;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Rate of additional treatment production (courses per day)",
                           "%lf", (void*)&(g_allParams.TreatNewCoursesRate), 1, 1, 0))
      g_allParams.TreatNewCoursesRate = 0;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                           "Maximum number of people targeted with radial prophylaxis per case", "%i",
                           (void*)&(g_allParams.TreatMaxCoursesPerCase), 1, 1, 0))
      g_allParams.TreatMaxCoursesPerCase = 1000000000;

   if (g_allParams.DoAdUnits)
   {
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Treat administrative units rather than rings", "%i",
                              (void*)&(g_allParams.TreatByAdminUnit), 1, 1, 0))
         g_allParams.TreatByAdminUnit = 0;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Administrative unit divisor for treatment", "%i",
                              (void*)&(g_allParams.TreatAdminUnitDivisor), 1, 1, 0))
         g_allParams.TreatAdminUnitDivisor = 1;
      if ((g_allParams.TreatAdminUnitDivisor == 0) || (g_allParams.TreatByAdminUnit == 0))
      {
         g_allParams.TreatByAdminUnit      = 0;
         g_allParams.TreatAdminUnitDivisor = 1;
      }
   }
   else
   {
      g_allParams.TreatAdminUnitDivisor = 1;
      g_allParams.TreatByAdminUnit      = 0;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Vaccination trigger incidence per cell", "%lf",
                           (void*)&(g_allParams.VaccCellIncThresh), 1, 1, 0))
      g_allParams.VaccCellIncThresh = 1000000000;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Relative susceptibility of vaccinated individual", "%lf",
                           (void*)&(g_allParams.VaccSuscDrop), 1, 1, 0))
      g_allParams.VaccSuscDrop = 1;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                           "Relative susceptibility of individual vaccinated after switch time", "%lf",
                           (void*)&(g_allParams.VaccSuscDrop2), 1, 1, 0))
      g_allParams.VaccSuscDrop2 = 1;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Switch time at which vaccine efficacy increases", "%lf",
                           (void*)&(g_allParams.VaccTimeEfficacySwitch), 1, 1, 0))
      g_allParams.VaccTimeEfficacySwitch = USHRT_MAX / g_allParams.TimeStepsPerDay;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Decay rate of vaccine efficacy (per year)", "%lf",
                           (void*)&(g_allParams.VaccEfficacyDecay), 1, 1, 0))
      g_allParams.VaccEfficacyDecay = 0;
   g_allParams.VaccEfficacyDecay /= DAYS_PER_YEAR;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Relative infectiousness of vaccinated individual", "%lf",
                           (void*)&(g_allParams.VaccInfDrop), 1, 1, 0))
      g_allParams.VaccInfDrop = 1;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                           "Proportion of symptomatic cases resulting in death prevented by vaccination", "%lf",
                           (void*)&(g_allParams.VaccMortDrop), 1, 1, 0))
      g_allParams.VaccMortDrop = 0;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Proportion of symptomatic cases prevented by vaccination",
                           "%lf", (void*)&(g_allParams.VaccSympDrop), 1, 1, 0))
      g_allParams.VaccSympDrop = 0;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Delay to vaccinate", "%lf",
                           (void*)&(g_allParams.VaccDelayMean), 1, 1, 0))
      g_allParams.VaccDelayMean = 0;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Delay from vaccination to full protection", "%lf",
                           (void*)&(g_allParams.VaccTimeToEfficacy), 1, 1, 0))
      g_allParams.VaccTimeToEfficacy = 0;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Years between rounds of vaccination", "%lf",
                           (void*)&(g_allParams.VaccCampaignInterval), 1, 1, 0))
      g_allParams.VaccCampaignInterval = 1e10;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Max vaccine doses per day", "%i",
                           (void*)&(g_allParams.VaccDosePerDay), 1, 1, 0))
      g_allParams.VaccDosePerDay = -1;
   g_allParams.VaccCampaignInterval *= DAYS_PER_YEAR;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Maximum number of rounds of vaccination", "%i",
                           (void*)&(g_allParams.VaccMaxRounds), 1, 1, 0))
      g_allParams.VaccMaxRounds = 1;
   if (g_allParams.DoHouseholds)
   {
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Proportion of households of cases vaccinated", "%lf",
                              (void*)&(g_allParams.VaccPropCaseHouseholds), 1, 1, 0))
         g_allParams.VaccPropCaseHouseholds = 0;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Duration of household vaccination policy", "%lf",
                              (void*)&(g_allParams.VaccHouseholdsDuration), 1, 1, 0))
         g_allParams.VaccHouseholdsDuration = USHRT_MAX / g_allParams.TimeStepsPerDay;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Vaccination start time", "%lf",
                           (void*)&(g_allParams.VaccTimeStartBase), 1, 1, 0))
      g_allParams.VaccTimeStartBase = USHRT_MAX / g_allParams.TimeStepsPerDay;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Proportion of population vaccinated", "%lf",
                           (void*)&(g_allParams.VaccProp), 1, 1, 0))
      g_allParams.VaccProp = 0;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Time taken to reach max vaccination coverage (in years)",
                           "%lf", (void*)&(g_allParams.VaccCoverageIncreasePeriod), 1, 1, 0))
      g_allParams.VaccCoverageIncreasePeriod = 0;
   g_allParams.VaccCoverageIncreasePeriod *= DAYS_PER_YEAR;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Time to start geographic vaccination", "%lf",
                           (void*)&(g_allParams.VaccTimeStartGeo), 1, 1, 0))
      g_allParams.VaccTimeStartGeo = 1e10;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Vaccination radius", "%lf",
                           (void*)&(g_allParams.VaccRadius), 1, 1, 0))
      g_allParams.VaccRadius = 0;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Minimum radius from case to vaccinate", "%lf",
                           (void*)&(g_allParams.VaccMinRadius), 1, 1, 0))
      g_allParams.VaccMinRadius = 0;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Maximum number of vaccine courses available", "%lf",
                           (void*)&(g_allParams.VaccMaxCoursesBase), 1, 1, 0))
      g_allParams.VaccMaxCoursesBase = 1e20;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Start time of additional vaccine production", "%lf",
                           (void*)&(g_allParams.VaccNewCoursesStartTime), 1, 1, 0))
      g_allParams.VaccNewCoursesStartTime = USHRT_MAX / g_allParams.TimeStepsPerDay;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "End time of additional vaccine production", "%lf",
                           (void*)&(g_allParams.VaccNewCoursesEndTime), 1, 1, 0))
      g_allParams.VaccNewCoursesEndTime = USHRT_MAX / g_allParams.TimeStepsPerDay;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Rate of additional vaccine production (courses per day)",
                           "%lf", (void*)&(g_allParams.VaccNewCoursesRate), 1, 1, 0))
      g_allParams.VaccNewCoursesRate = 0;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Apply mass rather than reactive vaccination", "%i",
                           (void*)&(g_allParams.DoMassVacc), 1, 1, 0))
      g_allParams.DoMassVacc = 0;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Priority age range for mass vaccination", "%i",
                           (void*)g_allParams.VaccPriorityGroupAge, 2, 1, 0))
   {
      g_allParams.VaccPriorityGroupAge[0] = 1;
      g_allParams.VaccPriorityGroupAge[1] = 0;
   }
   if (g_allParams.DoAdUnits)
   {
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Vaccinate administrative units rather than rings", "%i",
                              (void*)&(g_allParams.VaccByAdminUnit), 1, 1, 0))
         g_allParams.VaccByAdminUnit = 0;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Administrative unit divisor for vaccination", "%i",
                              (void*)&(g_allParams.VaccAdminUnitDivisor), 1, 1, 0))
         g_allParams.VaccAdminUnitDivisor = 1;
      if ((g_allParams.VaccAdminUnitDivisor == 0) || (g_allParams.VaccByAdminUnit == 0))
         g_allParams.VaccAdminUnitDivisor = 1;
   }
   else
   {
      g_allParams.VaccAdminUnitDivisor = 1;
      g_allParams.VaccByAdminUnit      = 0;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Movement restrictions trigger incidence per cell", "%i",
                           (void*)&(g_allParams.MoveRestrCellIncThresh), 1, 1, 0))
      g_allParams.MoveRestrCellIncThresh = 1000000000;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Delay to start movement restrictions", "%lf",
                           (void*)&(g_allParams.MoveDelayMean), 1, 1, 0))
      g_allParams.MoveDelayMean = 0;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Duration of movement restrictions", "%lf",
                           (void*)&(g_allParams.MoveRestrDuration), 1, 1, 0))
      g_allParams.MoveRestrDuration = 7;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Residual movements after restrictions", "%lf",
                           (void*)&(g_allParams.MoveRestrEffect), 1, 1, 0))
      g_allParams.MoveRestrEffect = 0;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Minimum radius of movement restrictions", "%lf",
                           (void*)&(g_allParams.MoveRestrRadius), 1, 1, 0))
      g_allParams.MoveRestrRadius = 0;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Movement restrictions start time", "%lf",
                           (void*)&(g_allParams.MoveRestrTimeStartBase), 1, 1, 0))
      g_allParams.MoveRestrTimeStartBase = USHRT_MAX / g_allParams.TimeStepsPerDay;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Impose blanket movement restrictions", "%i",
                           (void*)&(g_allParams.DoBlanketMoveRestr), 1, 1, 0))
      g_allParams.DoBlanketMoveRestr = 0;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Movement restrictions only once", "%i",
                           (void*)&(g_allParams.DoMoveRestrOnceOnly), 1, 1, 0))
      g_allParams.DoMoveRestrOnceOnly = 0;
   if (g_allParams.DoMoveRestrOnceOnly)
      g_allParams.DoMoveRestrOnceOnly = 4;
   if (g_allParams.DoAdUnits)
   {
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Movement restrictions in administrative units rather than rings", "%i",
                              (void*)&(g_allParams.MoveRestrByAdminUnit), 1, 1, 0))
         g_allParams.MoveRestrByAdminUnit = 0;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Administrative unit divisor for movement restrictions",
                              "%i", (void*)&(g_allParams.MoveRestrAdminUnitDivisor), 1, 1, 0))
         g_allParams.MoveRestrAdminUnitDivisor = 1;
      if ((g_allParams.MoveRestrAdminUnitDivisor == 0) || (g_allParams.MoveRestrByAdminUnit == 0))
         g_allParams.MoveRestrAdminUnitDivisor = 1;
   }
   else
   {
      g_allParams.MoveRestrAdminUnitDivisor = 1;
      g_allParams.MoveRestrByAdminUnit      = 0;
   }

   // Intervention delays and durations by admin unit: ggilani 16/03/20
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Include intervention delays by admin unit", "%i",
                           (void*)&(g_allParams.DoInterventionDelaysByAdUnit), 1, 1, 0))
      g_allParams.DoInterventionDelaysByAdUnit = 0;
   if (g_allParams.DoInterventionDelaysByAdUnit)
   {
      // Set up arrays to temporarily store parameters per admin unit
      double AdunitDelayToSocialDistance[MAX_ADUNITS];
      double AdunitDelayToHQuarantine[MAX_ADUNITS];
      double AdunitDelayToCaseIsolation[MAX_ADUNITS];
      double AdunitDelayToPlaceClose[MAX_ADUNITS];
      double AdunitDurationSocialDistance[MAX_ADUNITS];
      double AdunitDurationHQuarantine[MAX_ADUNITS];
      double AdunitDurationCaseIsolation[MAX_ADUNITS];
      double AdunitDurationPlaceClose[MAX_ADUNITS];

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Delay to social distancing by admin unit", "%lf",
                              (void*)AdunitDelayToSocialDistance, g_allParams.NumAdunits, 1, 0))
         for (int i = 0; i < g_allParams.NumAdunits; i++)
            AdunitDelayToSocialDistance[i] = 0;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Delay to household quarantine by admin unit", "%lf",
                              (void*)AdunitDelayToHQuarantine, g_allParams.NumAdunits, 1, 0))
         for (int i = 0; i < g_allParams.NumAdunits; i++)
            AdunitDelayToHQuarantine[i] = 0;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Delay to case isolation by admin unit", "%lf",
                              (void*)AdunitDelayToCaseIsolation, g_allParams.NumAdunits, 1, 0))
         for (int i = 0; i < g_allParams.NumAdunits; i++)
            AdunitDelayToCaseIsolation[i] = 0;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Delay to place closure by admin unit", "%lf",
                              (void*)AdunitDelayToPlaceClose, g_allParams.NumAdunits, 1, 0))
         for (int i = 0; i < g_allParams.NumAdunits; i++)
            AdunitDelayToPlaceClose[i] = 0;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Duration of social distancing by admin unit", "%lf",
                              (void*)AdunitDurationSocialDistance, g_allParams.NumAdunits, 1, 0))
         for (int i = 0; i < g_allParams.NumAdunits; i++)
            AdunitDurationSocialDistance[i] = 0;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Duration of household quarantine by admin unit", "%lf",
                              (void*)AdunitDurationHQuarantine, g_allParams.NumAdunits, 1, 0))
         for (int i = 0; i < g_allParams.NumAdunits; i++)
            AdunitDurationHQuarantine[i] = 0;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Duration of case isolation by admin unit", "%lf",
                              (void*)AdunitDurationCaseIsolation, g_allParams.NumAdunits, 1, 0))
         for (int i = 0; i < g_allParams.NumAdunits; i++)
            AdunitDurationCaseIsolation[i] = 0;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Duration of place closure by admin unit", "%lf",
                              (void*)AdunitDurationPlaceClose, g_allParams.NumAdunits, 1, 0))
         for (int i = 0; i < g_allParams.NumAdunits; i++)
            AdunitDurationPlaceClose[i] = 0;

      for (int i = 0; i < g_allParams.NumAdunits; i++)
      {
         AdUnits[i].SocialDistanceDelay    = AdunitDelayToSocialDistance[i];
         AdUnits[i].SocialDistanceDuration = AdunitDurationSocialDistance[i];
         AdUnits[i].HQuarantineDelay       = AdunitDelayToHQuarantine[i];
         AdUnits[i].HQuarantineDuration    = AdunitDurationHQuarantine[i];
         AdUnits[i].CaseIsolationDelay     = AdunitDelayToCaseIsolation[i];
         AdUnits[i].CaseIsolationDuration  = AdunitDurationCaseIsolation[i];
         AdUnits[i].PlaceCloseDelay        = AdunitDelayToPlaceClose[i];
         AdUnits[i].PlaceCloseDuration     = AdunitDurationPlaceClose[i];
      }
   }

   ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** /////
   ///**** ///// ****
   ///// **** DIGITAL CONTACT TRACING
   ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** /////
   ///**** ///// ****

   // New code for digital contact tracing - ggilani: 09/03/20
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Include digital contact tracing", "%i",
                           (void*)&(g_allParams.DoDigitalContactTracing), 1, 1, 0))
      g_allParams.DoDigitalContactTracing = 0;
   if (g_allParams.DoDigitalContactTracing)
   {
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Digital contact tracing trigger incidence per cell",
                              "%lf", (void*)&(g_allParams.DigitalContactTracing_CellIncThresh), 1, 1, 0))
         g_allParams.DigitalContactTracing_CellIncThresh = 1000000000;

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Proportion of population or households covered by digital contact tracing", "%lf",
                              (void*)&(g_allParams.PropPopUsingDigitalContactTracing), 1, 1, 0))
         g_allParams.PropPopUsingDigitalContactTracing = 1;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Proportion of smartphone users by age", "%lf",
                              (void*)g_allParams.ProportionSmartphoneUsersByAge, NUM_AGE_GROUPS, 1, 0))
      {
         for (int i = 0; i < NUM_AGE_GROUPS; i++)
         {
            g_allParams.ProportionSmartphoneUsersByAge[i] = 1;
         }
      }
      if (g_allParams.DoPlaces)
      {
         if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Cluster digital app clusters by household", "%i",
                                 (void*)&(g_allParams.ClusterDigitalContactUsers), 1, 1, 0))
            g_allParams.ClusterDigitalContactUsers = 0; // by default, don't cluster by location
      }
      else
      {
         g_allParams.ClusterDigitalContactUsers = 0;
      }
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Proportion of digital contacts who self-isolate", "%lf",
                              (void*)&(g_allParams.ProportionDigitalContactsIsolate), 1, 1, 0))
         g_allParams.ProportionDigitalContactsIsolate = 0;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Maximum number of contacts to trace per index case",
                              "%i", (void*)&(g_allParams.MaxDigitalContactsToTrace), 1, 1, 0))
         g_allParams.MaxDigitalContactsToTrace = MAX_CONTACTS;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Delay between isolation of index case and contacts",
                              "%lf", (void*)&(g_allParams.DigitalContactTracingDelay), 1, 1, 0))
         g_allParams.DigitalContactTracingDelay = g_allParams.TimeStep;
      // we really need one timestep between to make sure contact is not processed before index
      if (g_allParams.DigitalContactTracingDelay == 0)
         g_allParams.DigitalContactTracingDelay = g_allParams.TimeStep;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Length of self-isolation for digital contacts", "%lf",
                              (void*)&(g_allParams.LengthDigitalContactIsolation), 1, 1, 0))
         g_allParams.LengthDigitalContactIsolation = 0;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Spatial scaling factor - digital contact tracing",
                              "%lf", (void*)&(g_allParams.ScalingFactorSpatialDigitalContacts), 1, 1, 0))
         g_allParams.ScalingFactorSpatialDigitalContacts = 1;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Place scaling factor - digital contact tracing", "%lf",
                              (void*)&(g_allParams.ScalingFactorPlaceDigitalContacts), 1, 1, 0))
         g_allParams.ScalingFactorPlaceDigitalContacts = 1;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Digital contact tracing start time", "%lf",
                              (void*)&(g_allParams.DigitalContactTracingTimeStartBase), 1, 1, 0))
         g_allParams.DigitalContactTracingTimeStartBase = USHRT_MAX / g_allParams.TimeStepsPerDay;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Duration of digital contact tracing policy", "%lf",
                              (void*)&(g_allParams.DigitalContactTracingPolicyDuration), 1, 1, 0))
         g_allParams.DigitalContactTracingPolicyDuration = 7;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Output digital contact tracing", "%i",
                              (void*)&(g_allParams.OutputDigitalContactTracing), 1, 1, 0))
         g_allParams.OutputDigitalContactTracing = 0;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Output digital contact distribution", "%i",
                              (void*)&(g_allParams.OutputDigitalContactDist), 1, 1, 0))
         g_allParams.OutputDigitalContactDist = 0;

      // if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Include household contacts in digital contact
      // tracing", "%i", (void*) & (g_allParams.IncludeHouseholdDigitalContactTracing), 1, 1, 0))
      // g_allParams.IncludeHouseholdDigitalContactTracing = 1; if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
      // "Include place group contacts in digital contact tracing", "%i", (void*) &
      // (g_allParams.IncludePlaceGroupDigitalContactTracing), 1, 1, 0))
      // g_allParams.IncludePlaceGroupDigitalContactTracing = 1;

      // added admin unit specific delays by admin unit
      if (g_allParams.DoInterventionDelaysByAdUnit)
      {
         double AdunitDelayToDCT[MAX_ADUNITS];
         double AdunitDurationDCT[MAX_ADUNITS];

         if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Delay to digital contact tracing by admin unit",
                                 "%lf", (void*)AdunitDelayToDCT, g_allParams.NumAdunits, 1, 0))
         {
            for (int i = 0; i < g_allParams.NumAdunits; i++)
            {
               AdunitDelayToDCT[i] = 0;
            }
         }

         if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Duration of digital contact tracing by admin unit",
                                 "%lf", (void*)AdunitDurationDCT, g_allParams.NumAdunits, 1, 0))
         {
            for (int i = 0; i < g_allParams.NumAdunits; i++)
            {
               AdunitDurationDCT[i] = 0;
            }
         }

         for (int i = 0; i < g_allParams.NumAdunits; i++)
         {
            AdUnits[i].DCTDelay    = AdunitDelayToDCT[i];
            AdUnits[i].DCTDuration = AdunitDurationDCT[i];
         }
      }
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Isolate index cases in digital contact tracing", "%i",
                              (void*)&(g_allParams.DCTIsolateIndexCases), 1, 1, 0))
         g_allParams.DCTIsolateIndexCases = 1;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Residual contacts after digital contact tracing isolation", "%lf",
                              (void*)&(g_allParams.DCTCaseIsolationEffectiveness), 1, 1, 0))
         g_allParams.DCTCaseIsolationEffectiveness = g_allParams.CaseIsolationEffectiveness;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Residual household contacts after digital contact tracing isolation", "%lf",
                              (void*)&(g_allParams.DCTCaseIsolationHouseEffectiveness), 1, 1, 0))
         g_allParams.DCTCaseIsolationHouseEffectiveness = g_allParams.CaseIsolationHouseEffectiveness;
      // initialise total number of users to 0
      g_allParams.NDigitalContactUsers   = 0;
      g_allParams.NDigitalHouseholdUsers = 0;

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Delay between symptom onset and isolation for index case", "%lf",
                              (void*)&(g_allParams.DelayFromIndexCaseDetectionToDCTIsolation), 1, 1, 0))
         g_allParams.DelayFromIndexCaseDetectionToDCTIsolation = 0;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Test index cases and contacts", "%i",
                              (void*)&(g_allParams.DoDCTTest), 1, 1, 0))
         g_allParams.DoDCTTest = 0;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Delay to test index case", "%lf",
                              (void*)&(g_allParams.DelayToTestIndexCase), 1, 1, 0))
         g_allParams.DelayToTestIndexCase = 1;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Delay to test DCT contacts", "%lf",
                              (void*)&(g_allParams.DelayToTestDCTContacts), 1, 1, 0))
         g_allParams.DelayToTestDCTContacts = 7;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Testing specificity - DCT", "%lf",
                              (void*)&(g_allParams.SpecificityDCT), 1, 1, 0))
         g_allParams.SpecificityDCT = 1;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Testing sensitivity - DCT", "%lf",
                              (void*)&(g_allParams.SensitivityDCT), 1, 1, 0))
         g_allParams.SensitivityDCT = 1;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Find contacts of digital contacts", "%i",
                              (void*)&(g_allParams.FindContactsOfDCTContacts), 1, 1, 0))
         g_allParams.FindContactsOfDCTContacts = 0;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Remove contacts of a negative index case", "%i",
                              (void*)&(g_allParams.RemoveContactsOfNegativeIndexCase), 1, 1, 0))
         g_allParams.RemoveContactsOfNegativeIndexCase = 0;
   }
   else
   {
      // Set these to 1 so it doesn't interfere with code if we aren't using digital contact tracing.

      g_allParams.ScalingFactorSpatialDigitalContacts = 1;
      g_allParams.ScalingFactorPlaceDigitalContacts   = 1;
   }

   ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** /////
   ///**** ///// ****
   ///// **** PLACE CLOSURE
   ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** /////
   ///**** ///// ****

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Trigger incidence per cell for place closure", "%i",
                           (void*)&(g_allParams.PlaceCloseCellIncThresh1), 1, 1, 0))
      g_allParams.PlaceCloseCellIncThresh1 = 1000000000;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Trigger incidence per cell for second place closure", "%i",
                           (void*)&(g_allParams.PlaceCloseCellIncThresh2), 1, 1, 0))
      g_allParams.PlaceCloseCellIncThresh2 = 1000000000;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Trigger incidence per cell for end of place closure", "%i",
                           (void*)&(g_allParams.PlaceCloseCellIncStopThresh), 1, 1, 0))
      g_allParams.PlaceCloseCellIncStopThresh = 0;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Delay to start place closure", "%lf",
                           (void*)&(g_allParams.PlaceCloseDelayMean), 1, 1, 0))
      g_allParams.PlaceCloseDelayMean = 0;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Duration of place closure", "%lf",
                           (void*)&(g_allParams.PlaceCloseDurationBase), 1, 1, 0))
      g_allParams.PlaceCloseDurationBase = 7;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Duration of second place closure", "%lf",
                           (void*)&(g_allParams.PlaceCloseDuration2), 1, 1, 0))
      g_allParams.PlaceCloseDuration2 = 7;
   if (g_allParams.DoPlaces)
   {
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Proportion of places remaining open after closure by place type", "%lf",
                              (void*)g_allParams.PlaceCloseEffect, g_allParams.PlaceTypeNum, 1, 0))
         for (int i = 0; i < NUM_PLACE_TYPES; i++)
            g_allParams.PlaceCloseEffect[i] = 1;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Proportional attendance after closure by place type",
                              "%lf", (void*)g_allParams.PlaceClosePropAttending, g_allParams.PlaceTypeNum, 1, 0))
         for (int i = 0; i < NUM_PLACE_TYPES; i++)
            g_allParams.PlaceClosePropAttending[i] = 0;
   }
   if (g_allParams.DoHouseholds)
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Relative household contact rate after closure", "%lf",
                              (void*)&g_allParams.PlaceCloseHouseholdRelContact, 1, 1, 0))
         g_allParams.PlaceCloseHouseholdRelContact = 1;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Relative spatial contact rate after closure", "%lf",
                           (void*)&g_allParams.PlaceCloseSpatialRelContact, 1, 1, 0))
      g_allParams.PlaceCloseSpatialRelContact = 1;

   if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Include holidays", "%i", (void*)&(g_allParams.DoHolidays),
                           1, 1, 0))
      g_allParams.DoHolidays = 0;
   if (g_allParams.DoHolidays)
   {
      if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat,
                              "Proportion of places remaining open during holidays by place type", "%lf",
                              (void*)g_allParams.HolidayEffect, g_allParams.PlaceTypeNum, 1, 0))
      {
         for (int i = 0; i < NUM_PLACE_TYPES; i++)
         {
            g_allParams.HolidayEffect[i] = 1;
         }
      }

      if (!GetInputParameter2(PreParamFile_dat, AdminFile_dat, "Number of holidays", "%i",
                              (void*)&(g_allParams.NumHolidays), 1, 1, 0))
      {
         g_allParams.NumHolidays = 0;
      }

      if (g_allParams.NumHolidays > DAYS_PER_YEAR)
      {
         g_allParams.NumHolidays = DAYS_PER_YEAR;
      }

      if (g_allParams.NumHolidays > 0)
      {
         GetInputParameter(PreParamFile_dat, AdminFile_dat, "Holiday start times", "%lf",
                           (void*)g_allParams.HolidayStartTime, g_allParams.NumHolidays, 1, 0);
         GetInputParameter(PreParamFile_dat, AdminFile_dat, "Holiday durations", "%lf",
                           (void*)g_allParams.HolidayDuration, g_allParams.NumHolidays, 1, 0);
      }
   }
   else
   {
      g_allParams.NumHolidays = 0;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Minimum radius for place closure", "%lf",
                           (void*)&(g_allParams.PlaceCloseRadius), 1, 1, 0))
   {
      g_allParams.PlaceCloseRadius = 0;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Place closure start time", "%lf",
                           (void*)&(g_allParams.PlaceCloseTimeStartBase), 1, 1, 0))
      g_allParams.PlaceCloseTimeStartBase = USHRT_MAX / g_allParams.TimeStepsPerDay;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Place closure second start time", "%lf",
                           (void*)&(g_allParams.PlaceCloseTimeStartBase2), 1, 1, 0))
      g_allParams.PlaceCloseTimeStartBase2 = USHRT_MAX / g_allParams.TimeStepsPerDay;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Places close only once", "%i",
                           (void*)&(g_allParams.DoPlaceCloseOnceOnly), 1, 1, 0))
      g_allParams.DoPlaceCloseOnceOnly = 0;

   if (g_allParams.DoPlaceCloseOnceOnly)
      g_allParams.DoPlaceCloseOnceOnly = 4;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Place closure incidence threshold", "%i",
                           (void*)&(g_allParams.PlaceCloseIncTrig1), 1,
      1, 0))
      g_allParams.PlaceCloseIncTrig1 = 1;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Place closure second incidence threshold", "%i",
                           (void*)&(g_allParams.PlaceCloseIncTrig2), 1, 1, 0))
      g_allParams.PlaceCloseIncTrig2 = g_allParams.PlaceCloseIncTrig1;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Place closure fractional incidence threshold", "%lf",
                           (void*)&(g_allParams.PlaceCloseFracIncTrig), 1, 1, 0))
      g_allParams.PlaceCloseFracIncTrig = 0;

   if ((g_allParams.DoAdUnits) && (g_allParams.DoPlaces))
   {
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Place closure in administrative units rather than rings", "%i",
                              (void*)&(g_allParams.PlaceCloseByAdminUnit), 1, 1, 0))
         g_allParams.PlaceCloseByAdminUnit = 0;

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Administrative unit divisor for place closure", "%i",
                              (void*)&(g_allParams.PlaceCloseAdminUnitDivisor), 1, 1, 0))
         g_allParams.PlaceCloseAdminUnitDivisor = 1;

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Place types to close for admin unit closure (0/1 array)", "%i",
                              (void*)&(g_allParams.PlaceCloseAdunitPlaceTypes), g_allParams.PlaceTypeNum, 1, 0))
      {
         for (int i = 0; i < g_allParams.PlaceTypeNum; i++)
         {
            g_allParams.PlaceCloseAdunitPlaceTypes[i] = 0;
         }
      }
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Cumulative proportion of place members needing to become sick for admin unit closure",
                              "%lf", (void*)&(g_allParams.PlaceCloseCasePropThresh), 1, 1, 0))
         g_allParams.PlaceCloseCasePropThresh = 2;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Proportion of places in admin unit needing to pass threshold for place closure", "%lf",
                              (void*)&(g_allParams.PlaceCloseAdunitPropThresh), 1, 1, 0))
         g_allParams.PlaceCloseAdunitPropThresh = 2;
      if ((g_allParams.PlaceCloseAdminUnitDivisor < 1) || (g_allParams.PlaceCloseByAdminUnit == 0))
         g_allParams.PlaceCloseAdminUnitDivisor = 1;
   }
   else
   {
      g_allParams.PlaceCloseAdminUnitDivisor = 1;
      g_allParams.PlaceCloseByAdminUnit      = 0;
   }

   ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** /////
   ///**** ///// ****
   ///// **** SOCIAL DISTANCING
   ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** /////
   ///**** ///// ****

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Trigger incidence per cell for social distancing", "%i",
                           (void*)&(g_allParams.SocDistCellIncThresh), 1, 1, 0))
      g_allParams.SocDistCellIncThresh = 1000000000;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Trigger incidence per cell for end of social distancing",
                           "%i", (void*)&(g_allParams.SocDistCellIncStopThresh), 1, 1, 0))
      g_allParams.SocDistCellIncStopThresh = 0;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Duration of social distancing", "%lf",
                           (void*)&(g_allParams.SocDistDuration), 1, 1, 0))
      g_allParams.SocDistDuration = 7;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Duration of social distancing after change", "%lf",
                           (void*)&(g_allParams.SocDistDuration2), 1, 1, 0))
      g_allParams.SocDistDuration2 = 7;

   if (g_allParams.DoPlaces)
   {
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Relative place contact rate given social distancing by place type", "%lf",
                              (void*)g_allParams.SocDistPlaceEffect, g_allParams.PlaceTypeNum, 1, 0))
      {
         for (int i = 0; i < NUM_PLACE_TYPES; i++)
         {
            g_allParams.SocDistPlaceEffect[i] = 1;
         }
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Relative place contact rate given enhanced social distancing by place type", "%lf",
                              (void*)g_allParams.EnhancedSocDistPlaceEffect, g_allParams.PlaceTypeNum, 1, 0))
      {
         for (int i = 0; i < NUM_PLACE_TYPES; i++)
         {
            g_allParams.EnhancedSocDistPlaceEffect[i] = 1;
         }
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Relative place contact rate given social distancing by place type after change", "%lf",
                              (void*)g_allParams.SocDistPlaceEffect2, g_allParams.PlaceTypeNum, 1, 0))
      {
         for (int i = 0; i < NUM_PLACE_TYPES; i++)
         {
            g_allParams.SocDistPlaceEffect2[i] = g_allParams.SocDistPlaceEffect[i];
         }
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Relative place contact rate given enhanced social distancing by place type after change",
                              "%lf", (void*)g_allParams.EnhancedSocDistPlaceEffect2, g_allParams.PlaceTypeNum, 1, 0))
      {
         for (int i = 0; i < NUM_PLACE_TYPES; i++)
         {
            g_allParams.EnhancedSocDistPlaceEffect2[i] = g_allParams.EnhancedSocDistPlaceEffect[i];
            }
         }
   }

   if (g_allParams.DoHouseholds)
   {
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Relative household contact rate given social distancing", "%lf",
                              (void*)&g_allParams.SocDistHouseholdEffect, 1, 1, 0))
      {
         g_allParams.SocDistHouseholdEffect = 1;
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Relative household contact rate given enhanced social distancing", "%lf",
                              (void*)&g_allParams.EnhancedSocDistHouseholdEffect, 1, 1, 0))
      {
         g_allParams.EnhancedSocDistHouseholdEffect = 1;
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Relative household contact rate given social distancing  after change", "%lf",
                              (void*)&g_allParams.SocDistHouseholdEffect2, 1, 1, 0))
         g_allParams.SocDistHouseholdEffect2 = g_allParams.SocDistHouseholdEffect;

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Relative household contact rate given enhanced social distancing after change", "%lf",
                              (void*)&g_allParams.EnhancedSocDistHouseholdEffect2, 1, 1, 0))
         g_allParams.EnhancedSocDistHouseholdEffect2 = g_allParams.EnhancedSocDistHouseholdEffect;

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Cluster compliance with enhanced social distancing by household", "%i",
                              (void*)&g_allParams.EnhancedSocDistClusterByHousehold, 1, 1, 0))
         g_allParams.EnhancedSocDistClusterByHousehold = 0;
   }
   else
   {
      g_allParams.EnhancedSocDistClusterByHousehold = 0;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Relative spatial contact rate given social distancing",
                           "%lf", (void*)&g_allParams.SocDistSpatialEffect, 1, 1, 0))
      g_allParams.SocDistSpatialEffect = 1;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                           "Relative spatial contact rate given social distancing after change", "%lf",
                           (void*)&g_allParams.SocDistSpatialEffect2, 1, 1, 0))
      g_allParams.SocDistSpatialEffect2 = g_allParams.SocDistSpatialEffect;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Minimum radius for social distancing", "%lf",
                           (void*)&(g_allParams.SocDistRadius), 1, 1, 0))
      g_allParams.SocDistRadius = 0;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Social distancing start time", "%lf",
                           (void*)&(g_allParams.SocDistTimeStartBase), 1, 1, 0))
      g_allParams.SocDistTimeStartBase = USHRT_MAX / g_allParams.TimeStepsPerDay;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Delay for change in effectiveness of social distancing",
                           "%lf", (void*)&(g_allParams.SocDistChangeDelay), 1, 1, 0))
      g_allParams.SocDistChangeDelay = USHRT_MAX / g_allParams.TimeStepsPerDay;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                           "Proportion compliant with enhanced social distancing by age group", "%lf",
                           (void*)g_allParams.EnhancedSocDistProportionCompliant, NUM_AGE_GROUPS, 1, 0))
   {
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Proportion compliant with enhanced social distancing",
                              "%lf", (void*)&t, 1, 1, 0))
      {
         t = 0;
      }

      for (int i = 0; i < NUM_AGE_GROUPS; i++)
      {
         g_allParams.EnhancedSocDistProportionCompliant[i] = t;
      }
   }
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                           "Relative spatial contact rate given enhanced social distancing", "%lf",
                           (void*)&g_allParams.EnhancedSocDistSpatialEffect, 1, 1, 0))
      g_allParams.EnhancedSocDistSpatialEffect = 1;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                           "Relative spatial contact rate given enhanced social distancing after change", "%lf",
                           (void*)&g_allParams.EnhancedSocDistSpatialEffect2, 1, 1, 0))
      g_allParams.EnhancedSocDistSpatialEffect2 = g_allParams.EnhancedSocDistSpatialEffect;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Social distancing only once", "%i",
                           (void*)&(g_allParams.DoSocDistOnceOnly), 1, 1, 0))
      g_allParams.DoSocDistOnceOnly = 0;
   if (g_allParams.DoSocDistOnceOnly)
      g_allParams.DoSocDistOnceOnly = 4;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Airport closure effectiveness", "%lf",
                           (void*)&(g_allParams.AirportCloseEffectiveness), 1, 1, 0))
      g_allParams.AirportCloseEffectiveness = 0;
   g_allParams.AirportCloseEffectiveness = 1.0 - g_allParams.AirportCloseEffectiveness;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Airport closure start time", "%lf",
                           (void*)&(g_allParams.AirportCloseTimeStartBase), 1, 1, 0))
      g_allParams.AirportCloseTimeStartBase = USHRT_MAX / g_allParams.TimeStepsPerDay;
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Airport closure duration", "%lf",
                           (void*)&(g_allParams.AirportCloseDuration), 1, 1, 0))
      g_allParams.AirportCloseDuration = USHRT_MAX / g_allParams.TimeStepsPerDay;

   ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** /////
   ///**** ///// ****
   ///// **** HOUSEHOLD QUARANTINE
   ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** /////
   ///**** ///// ****

   if (g_allParams.DoHouseholds)
   {
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Retrigger household quarantine with each new case in quarantine window", "%i",
                              (void*)&(g_allParams.DoHQretrigger), 1, 1, 0))
         g_allParams.DoHQretrigger = 0;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Household quarantine start time", "%lf",
                              (void*)&(g_allParams.HQuarantineTimeStartBase), 1, 1, 0))
         g_allParams.HQuarantineTimeStartBase = USHRT_MAX / g_allParams.TimeStepsPerDay;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Delay to start household quarantine", "%lf",
                              (void*)&(g_allParams.HQuarantineHouseDelay), 1, 1, 0))
         g_allParams.HQuarantineHouseDelay = 0;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Length of time households are quarantined", "%lf",
                              (void*)&(g_allParams.HQuarantineHouseDuration), 1, 1, 0))
         g_allParams.HQuarantineHouseDuration = 0;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Duration of household quarantine policy", "%lf",
                              (void*)&(g_allParams.HQuarantinePolicyDuration), 1, 1, 0))
         g_allParams.HQuarantinePolicyDuration = USHRT_MAX / g_allParams.TimeStepsPerDay;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Relative household contact rate after quarantine",
                              "%lf", (void*)&(g_allParams.HQuarantineHouseEffect), 1, 1, 0))
         g_allParams.HQuarantineHouseEffect = 1;
      if (g_allParams.DoPlaces)
      {
         if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                                 "Residual place contacts after household quarantine by place type", "%lf",
                                 (void*)g_allParams.HQuarantinePlaceEffect, g_allParams.PlaceTypeNum, 1, 0))
         {
            for (int i = 0; i < NUM_PLACE_TYPES; i++)
            {
               g_allParams.HQuarantinePlaceEffect[i] = 1;
            }
         }
      }
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Residual spatial contacts after household quarantine",
                              "%lf", (void*)&(g_allParams.HQuarantineSpatialEffect), 1, 1, 0))
         g_allParams.HQuarantineSpatialEffect = 1;

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Household level compliance with quarantine", "%lf",
                              (void*)&(g_allParams.HQuarantinePropHouseCompliant), 1, 1, 0))
         g_allParams.HQuarantinePropHouseCompliant = 1;

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Individual level compliance with quarantine", "%lf",
                              (void*)&(g_allParams.HQuarantinePropIndivCompliant), 1, 1, 0))
         g_allParams.HQuarantinePropIndivCompliant = 1;
   }
   else
   {
      g_allParams.HQuarantineTimeStartBase = 1e10;
   }

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Case isolation start time", "%lf",
                           (void*)&(g_allParams.CaseIsolationTimeStartBase), 1, 1, 0))
      g_allParams.CaseIsolationTimeStartBase = USHRT_MAX / g_allParams.TimeStepsPerDay;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Proportion of detected cases isolated", "%lf",
                           (void*)&(g_allParams.CaseIsolationProp), 1, 1, 0))
      g_allParams.CaseIsolationProp = 0;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Delay to start case isolation", "%lf",
                           (void*)&(g_allParams.CaseIsolationDelay), 1, 1, 0))
      g_allParams.CaseIsolationDelay = 0;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Duration of case isolation", "%lf",
                           (void*)&(g_allParams.CaseIsolationDuration), 1, 1, 0))
      g_allParams.CaseIsolationDuration = 0;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Duration of case isolation policy", "%lf",
                           (void*)&(g_allParams.CaseIsolationPolicyDuration), 1, 1, 0))
      g_allParams.CaseIsolationPolicyDuration = 1e10;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Residual contacts after case isolation", "%lf",
                           (void*)&(g_allParams.CaseIsolationEffectiveness), 1, 1, 0))
      g_allParams.CaseIsolationEffectiveness = 1;

   if (g_allParams.DoHouseholds)
   {
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Residual household contacts after case isolation",
                              "%lf", (void*)&(g_allParams.CaseIsolationHouseEffectiveness), 1, 1, 0))
         g_allParams.CaseIsolationHouseEffectiveness = g_allParams.CaseIsolationEffectiveness;
   }

   ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** /////
   ///**** ///// ****
   ///// **** VARIABLE EFFICACIES OVER TIME
   ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** ///// **** /////
   ///**** ///// ****

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Vary efficacies over time", "%i",
                           (void*)&(g_allParams.VaryEfficaciesOverTime), 1, 1, 0))
      g_allParams.VaryEfficaciesOverTime = 0;
   //// **** number of change times
   if (!g_allParams.VaryEfficaciesOverTime)
   {
      g_allParams.Num_SD_ChangeTimes  = 1;
      g_allParams.Num_CI_ChangeTimes  = 1;
      g_allParams.Num_HQ_ChangeTimes  = 1;
      g_allParams.Num_PC_ChangeTimes  = 1;
      g_allParams.Num_DCT_ChangeTimes = 1;
   }
   else
   {
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Number of change times for levels of social distancing",
                              "%i", (void*)&(g_allParams.Num_SD_ChangeTimes), 1, 1, 0))
         g_allParams.Num_SD_ChangeTimes = 1;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Number of change times for levels of case isolation",
                              "%i", (void*)&(g_allParams.Num_CI_ChangeTimes), 1, 1, 0))
         g_allParams.Num_CI_ChangeTimes = 1;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Number of change times for levels of household quarantine", "%i",
                              (void*)&(g_allParams.Num_HQ_ChangeTimes), 1, 1, 0))
         g_allParams.Num_HQ_ChangeTimes = 1;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Number of change times for levels of place closure",
                              "%i", (void*)&(g_allParams.Num_PC_ChangeTimes), 1, 1, 0))
         g_allParams.Num_PC_ChangeTimes = 1;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Number of change times for levels of digital contact tracing", "%i",
                              (void*)&(g_allParams.Num_DCT_ChangeTimes), 1, 1, 0))
         g_allParams.Num_DCT_ChangeTimes = 1;
   }

   //// **** change times:
   //// By default, initialize first change time to zero and all subsequent change times to occur after simulation time,
   ///i.e. single value of efficacy for social distancing.
   g_allParams.SD_ChangeTimes[0]  = 0;
   g_allParams.CI_ChangeTimes[0]  = 0;
   g_allParams.HQ_ChangeTimes[0]  = 0;
   g_allParams.PC_ChangeTimes[0]  = 0;
   g_allParams.DCT_ChangeTimes[0] = 0;
   for (int ChangeTime = 1; ChangeTime < MAX_NUM_INTERVENTION_CHANGE_TIMES; ChangeTime++)
   {
      g_allParams.SD_ChangeTimes[ChangeTime]  = 1e10;
      g_allParams.CI_ChangeTimes[ChangeTime]  = 1e10;
      g_allParams.HQ_ChangeTimes[ChangeTime]  = 1e10;
      g_allParams.PC_ChangeTimes[ChangeTime]  = 1e10;
      g_allParams.DCT_ChangeTimes[ChangeTime] = 1e10;
   }
   //// Get real values from (pre)param file
   GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Change times for levels of social distancing", "%lf",
                      (void*)g_allParams.SD_ChangeTimes, g_allParams.Num_SD_ChangeTimes, 1, 0);
   GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Change times for levels of case isolation", "%lf",
                      (void*)g_allParams.CI_ChangeTimes, g_allParams.Num_CI_ChangeTimes, 1, 0);
   GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Change times for levels of household quarantine", "%lf",
                      (void*)g_allParams.HQ_ChangeTimes, g_allParams.Num_HQ_ChangeTimes, 1, 0);
   GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Change times for levels of place closure", "%lf",
                      (void*)g_allParams.PC_ChangeTimes, g_allParams.Num_PC_ChangeTimes, 1, 0);
   GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Change times for levels of digital contact tracing", "%lf",
                      (void*)g_allParams.DCT_ChangeTimes, g_allParams.Num_DCT_ChangeTimes, 1, 0);

   // initialize to zero (regardless of whether doing places or households).
   for (int ChangeTime = 0; ChangeTime < MAX_NUM_INTERVENTION_CHANGE_TIMES; ChangeTime++)
   {
      //// **** "efficacies"
      //// spatial
      g_allParams.SD_SpatialEffects_OverTime[ChangeTime]          = 0;
      g_allParams.Enhanced_SD_SpatialEffects_OverTime[ChangeTime] = 0;
      g_allParams.CI_SpatialAndPlaceEffects_OverTime[ChangeTime]  = 0;
      g_allParams.HQ_SpatialEffects_OverTime[ChangeTime]          = 0;
      g_allParams.PC_SpatialEffects_OverTime[ChangeTime]          = 0;
      g_allParams.DCT_SpatialAndPlaceEffects_OverTime[ChangeTime] = 0;

      //// Household
      g_allParams.SD_HouseholdEffects_OverTime[ChangeTime]          = 0;
      g_allParams.Enhanced_SD_HouseholdEffects_OverTime[ChangeTime] = 0;
      g_allParams.CI_HouseholdEffects_OverTime[ChangeTime]          = 0;
      g_allParams.HQ_HouseholdEffects_OverTime[ChangeTime]          = 0;
      g_allParams.PC_HouseholdEffects_OverTime[ChangeTime]          = 0;
      g_allParams.DCT_HouseholdEffects_OverTime[ChangeTime]         = 0;

      //// place
      for (int PlaceType = 0; PlaceType < g_allParams.PlaceTypeNum; PlaceType++)
      {
         g_allParams.SD_PlaceEffects_OverTime[ChangeTime][PlaceType]          = 0;
         g_allParams.Enhanced_SD_PlaceEffects_OverTime[ChangeTime][PlaceType] = 0;
         g_allParams.HQ_PlaceEffects_OverTime[ChangeTime][PlaceType]          = 0;
         g_allParams.PC_PlaceEffects_OverTime[ChangeTime][PlaceType]          = 0;
      }
      g_allParams.PC_Durs_OverTime[ChangeTime] = 0;

      //// **** compliance
      g_allParams.CI_Prop_OverTime[ChangeTime]                  = 0;
      g_allParams.HQ_Individual_PropComply_OverTime[ChangeTime] = 0;
      g_allParams.HQ_Household_PropComply_OverTime[ChangeTime]  = 0;
      g_allParams.DCT_Prop_OverTime[ChangeTime]                 = 0;
   }

   //// **** "efficacies": by default, initialize to values read in previously.
   ///// spatial contact rates rates over time (and place too for CI and DCT)
   //// soc dist
   if (!g_allParams.VaryEfficaciesOverTime
       || !GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Relative spatial contact rates over time given social distancing", "%lf",
                              (void*)g_allParams.SD_SpatialEffects_OverTime, g_allParams.Num_SD_ChangeTimes, 1, 0))
      for (int ChangeTime = 0; ChangeTime < g_allParams.Num_SD_ChangeTimes; ChangeTime++)
         g_allParams.SD_SpatialEffects_OverTime[ChangeTime] =
            g_allParams.SocDistSpatialEffect; //// by default, initialize to Relative spatial contact rate given social
                                              ///distancing
   //// enhanced soc dist
   if (!g_allParams.VaryEfficaciesOverTime
       || !GetInputParameter2(
          ParamFile_dat, PreParamFile_dat, "Relative spatial contact rates over time given enhanced social distancing",
          "%lf", (void*)g_allParams.Enhanced_SD_SpatialEffects_OverTime, g_allParams.Num_SD_ChangeTimes, 1, 0))
      for (int ChangeTime = 0; ChangeTime < g_allParams.Num_SD_ChangeTimes; ChangeTime++)
         g_allParams.Enhanced_SD_SpatialEffects_OverTime[ChangeTime] =
            g_allParams.EnhancedSocDistSpatialEffect; //// by default, initialize to Relative spatial contact rate given
                                                      ///enhanced social distancing
   //// case isolation
   if (!g_allParams.VaryEfficaciesOverTime
       || !GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Residual contacts after case isolation over time",
                              "%lf", (void*)g_allParams.CI_SpatialAndPlaceEffects_OverTime,
                              g_allParams.Num_CI_ChangeTimes, 1, 0))
      for (int ChangeTime = 0; ChangeTime < g_allParams.Num_CI_ChangeTimes; ChangeTime++)
         g_allParams.CI_SpatialAndPlaceEffects_OverTime[ChangeTime] = g_allParams.CaseIsolationEffectiveness;
   //// household quarantine
   if (!g_allParams.VaryEfficaciesOverTime
       || !GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Residual spatial contacts over time after household quarantine", "%lf",
                              (void*)g_allParams.HQ_SpatialEffects_OverTime, g_allParams.Num_HQ_ChangeTimes, 1, 0))
      for (int ChangeTime = 0; ChangeTime < g_allParams.Num_HQ_ChangeTimes; ChangeTime++)
         g_allParams.HQ_SpatialEffects_OverTime[ChangeTime] = g_allParams.HQuarantineSpatialEffect;
   //// place closure
   if (!g_allParams.VaryEfficaciesOverTime
       || !GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Relative spatial contact rates over time after place closure", "%lf",
                              (void*)g_allParams.PC_SpatialEffects_OverTime, g_allParams.Num_PC_ChangeTimes, 1, 0))
      for (int ChangeTime = 0; ChangeTime < g_allParams.Num_PC_ChangeTimes; ChangeTime++)
         g_allParams.PC_SpatialEffects_OverTime[ChangeTime] = g_allParams.PlaceCloseSpatialRelContact;
   //// digital contact tracing
   if (!g_allParams.VaryEfficaciesOverTime
       || !GetInputParameter2(
          ParamFile_dat, PreParamFile_dat, "Residual contacts after digital contact tracing isolation over time", "%lf",
          (void*)g_allParams.DCT_SpatialAndPlaceEffects_OverTime, g_allParams.Num_DCT_ChangeTimes, 1, 0))
      for (int ChangeTime = 0; ChangeTime < g_allParams.Num_DCT_ChangeTimes; ChangeTime++)
         g_allParams.DCT_SpatialAndPlaceEffects_OverTime[ChangeTime] = g_allParams.DCTCaseIsolationEffectiveness;

   ///// Household contact rates over time
   if (g_allParams.DoHouseholds)
   {
      //// soc dist
      if (!g_allParams.VaryEfficaciesOverTime
          || !GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                                 "Relative household contact rates over time given social distancing", "%lf",
                                 (void*)g_allParams.SD_HouseholdEffects_OverTime, g_allParams.Num_SD_ChangeTimes, 1, 0))
         for (int ChangeTime = 0; ChangeTime < g_allParams.Num_SD_ChangeTimes; ChangeTime++)
            g_allParams.SD_HouseholdEffects_OverTime[ChangeTime] = g_allParams.SocDistHouseholdEffect;
      //// enhanced soc dist
      if (!g_allParams.VaryEfficaciesOverTime
          || !GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                                 "Relative household contact rates over time given enhanced social distancing", "%lf",
                                 (void*)g_allParams.Enhanced_SD_HouseholdEffects_OverTime,
                                 g_allParams.Num_SD_ChangeTimes, 1, 0))
         for (int ChangeTime = 0; ChangeTime < g_allParams.Num_SD_ChangeTimes; ChangeTime++)
            g_allParams.Enhanced_SD_HouseholdEffects_OverTime[ChangeTime] = g_allParams.EnhancedSocDistHouseholdEffect;
      //// case isolation
      if (!g_allParams.VaryEfficaciesOverTime
          || !GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                                 "Residual household contacts after case isolation over time", "%lf",
                                 (void*)g_allParams.CI_HouseholdEffects_OverTime, g_allParams.Num_CI_ChangeTimes, 1, 0))
         for (int ChangeTime = 0; ChangeTime < g_allParams.Num_CI_ChangeTimes; ChangeTime++)
            g_allParams.CI_HouseholdEffects_OverTime[ChangeTime] = g_allParams.CaseIsolationHouseEffectiveness;
      //// household quarantine
      if (!g_allParams.VaryEfficaciesOverTime
          || !GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                                 "Relative household contact rates over time after quarantine", "%lf",
                                 (void*)g_allParams.HQ_HouseholdEffects_OverTime, g_allParams.Num_HQ_ChangeTimes, 1, 0))
         for (int ChangeTime = 0; ChangeTime < g_allParams.Num_HQ_ChangeTimes; ChangeTime++)
            g_allParams.HQ_HouseholdEffects_OverTime[ChangeTime] = g_allParams.HQuarantineHouseEffect;
      //// place closure
      if (!g_allParams.VaryEfficaciesOverTime
          || !GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                                 "Relative household contact rates over time after place closure", "%lf",
                                 (void*)g_allParams.PC_HouseholdEffects_OverTime, g_allParams.Num_PC_ChangeTimes, 1, 0))
         for (int ChangeTime = 0; ChangeTime < g_allParams.Num_PC_ChangeTimes; ChangeTime++)
            g_allParams.PC_HouseholdEffects_OverTime[ChangeTime] = g_allParams.PlaceCloseHouseholdRelContact;
      //// digital contact tracing
      if (!g_allParams.VaryEfficaciesOverTime
          || !GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                                 "Residual household contacts after digital contact tracing isolation over time", "%lf",
                                 (void*)g_allParams.DCT_HouseholdEffects_OverTime, g_allParams.Num_DCT_ChangeTimes, 1,
                                 0))
         for (int ChangeTime = 0; ChangeTime < g_allParams.Num_DCT_ChangeTimes; ChangeTime++)
            g_allParams.DCT_HouseholdEffects_OverTime[ChangeTime] = g_allParams.DCTCaseIsolationHouseEffectiveness;
   }

   ///// place contact rates over time
   if (g_allParams.DoPlaces)
   {
      //// soc dist
      if (!g_allParams.VaryEfficaciesOverTime
          || !GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                                 "Relative place contact rates over time given social distancing by place type", "%lf",
                                 (void*)&g_allParams.SD_PlaceEffects_OverTime[0][0],
                                 g_allParams.Num_SD_ChangeTimes * g_allParams.PlaceTypeNum, 1, 0))
         for (int ChangeTime = 0; ChangeTime < g_allParams.Num_SD_ChangeTimes;
              ChangeTime++) //// by default populate to values of g_allParams.SocDistPlaceEffect
            for (int PlaceType = 0; PlaceType < g_allParams.PlaceTypeNum; PlaceType++)
               g_allParams.SD_PlaceEffects_OverTime[ChangeTime][PlaceType] = g_allParams.SocDistPlaceEffect[PlaceType];

      //// enhanced soc dist
      if (!g_allParams.VaryEfficaciesOverTime
          || !GetInputParameter2(
             ParamFile_dat, PreParamFile_dat,
             "Relative place contact rates over time given enhanced social distancing by place type", "%lf",
             (void*)&g_allParams.Enhanced_SD_PlaceEffects_OverTime[0][0],
             g_allParams.Num_SD_ChangeTimes * g_allParams.PlaceTypeNum, 1, 0))
         for (int ChangeTime = 0; ChangeTime < g_allParams.Num_SD_ChangeTimes;
              ChangeTime++) //// by default populate to values of g_allParams.EnhancedSocDistPlaceEffect
            for (int PlaceType = 0; PlaceType < g_allParams.PlaceTypeNum; PlaceType++)
               g_allParams.Enhanced_SD_PlaceEffects_OverTime[ChangeTime][PlaceType] =
                  g_allParams.EnhancedSocDistPlaceEffect[PlaceType];

      //// household quarantine
      if (!g_allParams.VaryEfficaciesOverTime
          || !GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                                 "Residual place contacts over time after household quarantine by place type", "%lf",
                                 (void*)&g_allParams.HQ_PlaceEffects_OverTime[0][0],
                                 g_allParams.Num_HQ_ChangeTimes * g_allParams.PlaceTypeNum, 1, 0))
         for (int ChangeTime = 0; ChangeTime < g_allParams.Num_HQ_ChangeTimes;
              ChangeTime++) //// by default populate to values of g_allParams.HQuarantinePlaceEffect
            for (int PlaceType = 0; PlaceType < g_allParams.PlaceTypeNum; PlaceType++)
               g_allParams.HQ_PlaceEffects_OverTime[ChangeTime][PlaceType] =
                  g_allParams.HQuarantinePlaceEffect[PlaceType];

      //// place closure
      if (!g_allParams.VaryEfficaciesOverTime
          || !GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                                 "Proportion of places remaining open after closure by place type over time", "%lf",
                                 (void*)&g_allParams.PC_PlaceEffects_OverTime[0][0],
                                 g_allParams.Num_PC_ChangeTimes * g_allParams.PlaceTypeNum, 1, 0))
         for (int ChangeTime = 0; ChangeTime < g_allParams.Num_PC_ChangeTimes;
              ChangeTime++) //// by default populate to values of g_allParams.PlaceCloseEffect
            for (int PlaceType = 0; PlaceType < g_allParams.PlaceTypeNum; PlaceType++)
               g_allParams.PC_PlaceEffects_OverTime[ChangeTime][PlaceType] = g_allParams.PlaceCloseEffect[PlaceType];

      if (!g_allParams.VaryEfficaciesOverTime
          || !GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                                 "Proportional attendance after closure by place type over time", "%lf",
                                 (void*)&g_allParams.PC_PropAttending_OverTime[0][0],
                                 g_allParams.Num_PC_ChangeTimes * g_allParams.PlaceTypeNum, 1, 0))
         for (int ChangeTime = 0; ChangeTime < g_allParams.Num_PC_ChangeTimes;
              ChangeTime++) //// by default populate to values of g_allParams.PlaceClosePropAttending
            for (int PlaceType = 0; PlaceType < g_allParams.PlaceTypeNum; PlaceType++)
               g_allParams.PC_PropAttending_OverTime[ChangeTime][PlaceType] =
                  g_allParams.PlaceClosePropAttending[PlaceType];
   }

   //// ****  compliance
   //// case isolation
   if (!g_allParams.VaryEfficaciesOverTime
       || !GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Proportion of detected cases isolated over time", "%lf",
                              (void*)g_allParams.CI_Prop_OverTime, g_allParams.Num_CI_ChangeTimes, 1, 0))
      for (int ChangeTime = 0; ChangeTime < g_allParams.Num_CI_ChangeTimes; ChangeTime++)
         g_allParams.CI_Prop_OverTime[ChangeTime] = g_allParams.CaseIsolationProp;
   //// household quarantine (individual level)
   if (!g_allParams.VaryEfficaciesOverTime
       || !GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Individual level compliance with quarantine over time",
                              "%lf", (void*)g_allParams.HQ_Individual_PropComply_OverTime,
                              g_allParams.Num_HQ_ChangeTimes, 1, 0))
      for (int ChangeTime = 0; ChangeTime < g_allParams.Num_HQ_ChangeTimes; ChangeTime++)
         g_allParams.HQ_Individual_PropComply_OverTime[ChangeTime] = g_allParams.HQuarantinePropIndivCompliant;
   //// household quarantine (Household level)
   if (!g_allParams.VaryEfficaciesOverTime
       || !GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Household level compliance with quarantine over time",
                              "%lf", (void*)g_allParams.HQ_Household_PropComply_OverTime,
                              g_allParams.Num_HQ_ChangeTimes, 1, 0))
      for (int ChangeTime = 0; ChangeTime < g_allParams.Num_HQ_ChangeTimes; ChangeTime++)
         g_allParams.HQ_Household_PropComply_OverTime[ChangeTime] = g_allParams.HQuarantinePropHouseCompliant;
   //// digital contact tracing
   if (!g_allParams.VaryEfficaciesOverTime
       || !GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Proportion of digital contacts who self-isolate over time", "%lf",
                              (void*)g_allParams.DCT_Prop_OverTime, g_allParams.Num_DCT_ChangeTimes, 1, 0))
      for (int ChangeTime = 0; ChangeTime < g_allParams.Num_DCT_ChangeTimes; ChangeTime++)
         g_allParams.DCT_Prop_OverTime[ChangeTime] = g_allParams.ProportionDigitalContactsIsolate;
   if (!g_allParams.VaryEfficaciesOverTime
       || !GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Maximum number of contacts to trace per index case over time", "%i",
                              (void*)g_allParams.DCT_MaxToTrace_OverTime, g_allParams.Num_DCT_ChangeTimes, 1, 0))
      for (int ChangeTime = 0; ChangeTime < g_allParams.Num_DCT_ChangeTimes; ChangeTime++)
         g_allParams.DCT_MaxToTrace_OverTime[ChangeTime] = g_allParams.MaxDigitalContactsToTrace;
   if (g_allParams.DoPlaces)
   {
      //// ****  thresholds
      //// place closure (global threshold)
      if (!g_allParams.VaryEfficaciesOverTime
          || !GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Place closure incidence threshold over time", "%lf",
                                 (void*)g_allParams.PC_IncThresh_OverTime, g_allParams.Num_PC_ChangeTimes, 1, 0))
         for (int ChangeTime = 0; ChangeTime < g_allParams.Num_PC_ChangeTimes; ChangeTime++)
            g_allParams.PC_IncThresh_OverTime[ChangeTime] = g_allParams.PlaceCloseIncTrig1;
      //// place closure (fractional global threshold)
      if (!g_allParams.VaryEfficaciesOverTime
          || !GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                                 "Place closure fractional incidence threshold over time", "%lf",
                                 (void*)g_allParams.PC_FracIncThresh_OverTime, g_allParams.Num_PC_ChangeTimes, 1, 0))
         for (int ChangeTime = 0; ChangeTime < g_allParams.Num_PC_ChangeTimes; ChangeTime++)
            g_allParams.PC_FracIncThresh_OverTime[ChangeTime] = g_allParams.PlaceCloseFracIncTrig;
      //// place closure (cell incidence threshold)
      if (!g_allParams.VaryEfficaciesOverTime
          || !GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                                 "Trigger incidence per cell for place closure over time", "%i",
                                 (void*)g_allParams.PC_CellIncThresh_OverTime, g_allParams.Num_PC_ChangeTimes, 1, 0))
         for (int ChangeTime = 0; ChangeTime < g_allParams.Num_PC_ChangeTimes; ChangeTime++)
            g_allParams.PC_CellIncThresh_OverTime[ChangeTime] = g_allParams.PlaceCloseCellIncThresh1;
   }
   //// household quarantine
   if (!g_allParams.VaryEfficaciesOverTime
       || !GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Household quarantine trigger incidence per cell over time", "%lf",
                              (void*)g_allParams.HQ_CellIncThresh_OverTime, g_allParams.Num_HQ_ChangeTimes, 1, 0))
      for (int ChangeTime = 0; ChangeTime < g_allParams.Num_HQ_ChangeTimes; ChangeTime++)
         g_allParams.HQ_CellIncThresh_OverTime[ChangeTime] = g_allParams.HHQuar_CellIncThresh;
   //// case isolation
   if (!g_allParams.VaryEfficaciesOverTime
       || !GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Case isolation trigger incidence per cell over time",
                              "%lf", (void*)g_allParams.CI_CellIncThresh_OverTime, g_allParams.Num_CI_ChangeTimes, 1,
                              0))
      for (int ChangeTime = 0; ChangeTime < g_allParams.Num_CI_ChangeTimes; ChangeTime++)
         g_allParams.CI_CellIncThresh_OverTime[ChangeTime] = g_allParams.CaseIsolation_CellIncThresh;
   //// soc dists
   if (!g_allParams.VaryEfficaciesOverTime
       || !GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Trigger incidence per cell for social distancing over time", "%i",
                              (void*)g_allParams.SD_CellIncThresh_OverTime, g_allParams.Num_SD_ChangeTimes, 1, 0))
      for (int ChangeTime = 0; ChangeTime < g_allParams.Num_SD_ChangeTimes; ChangeTime++)
         g_allParams.SD_CellIncThresh_OverTime[ChangeTime] = g_allParams.SocDistCellIncThresh;

   //// **** Durations (later add Case isolation and Household quarantine)
   // place closure
   if (!g_allParams.VaryEfficaciesOverTime
       || !GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Duration of place closure over time", "%lf",
                              (void*)g_allParams.PC_Durs_OverTime, g_allParams.Num_PC_ChangeTimes, 1, 0))
      for (int ChangeTime = 0; ChangeTime < g_allParams.Num_PC_ChangeTimes; ChangeTime++)
         g_allParams.PC_Durs_OverTime[ChangeTime] = g_allParams.PlaceCloseDurationBase;

   //// Guards: make unused change values in array equal to final used value
   if (g_allParams.VaryEfficaciesOverTime)
   {
      //// soc dist
      for (int SD_ChangeTime = g_allParams.Num_SD_ChangeTimes; SD_ChangeTime < MAX_NUM_INTERVENTION_CHANGE_TIMES - 1;
           SD_ChangeTime++)
      {
         //// non-enhanced
         g_allParams.SD_SpatialEffects_OverTime[SD_ChangeTime] =
            g_allParams.SD_SpatialEffects_OverTime[g_allParams.Num_SD_ChangeTimes - 1];
         g_allParams.SD_HouseholdEffects_OverTime[SD_ChangeTime] =
            g_allParams.SD_HouseholdEffects_OverTime[g_allParams.Num_SD_ChangeTimes - 1];
         for (int PlaceType = 0; PlaceType < g_allParams.PlaceTypeNum; PlaceType++)
            g_allParams.SD_PlaceEffects_OverTime[SD_ChangeTime][PlaceType] =
               g_allParams.SD_PlaceEffects_OverTime[g_allParams.Num_SD_ChangeTimes - 1][PlaceType];
         //// enhanced
         g_allParams.Enhanced_SD_SpatialEffects_OverTime[SD_ChangeTime] =
            g_allParams.Enhanced_SD_SpatialEffects_OverTime[g_allParams.Num_SD_ChangeTimes - 1];
         g_allParams.Enhanced_SD_HouseholdEffects_OverTime[SD_ChangeTime] =
            g_allParams.Enhanced_SD_HouseholdEffects_OverTime[g_allParams.Num_SD_ChangeTimes - 1];
         for (int PlaceType = 0; PlaceType < g_allParams.PlaceTypeNum; PlaceType++)
            g_allParams.Enhanced_SD_PlaceEffects_OverTime[SD_ChangeTime][PlaceType] =
               g_allParams.Enhanced_SD_PlaceEffects_OverTime[g_allParams.Num_SD_ChangeTimes - 1][PlaceType];

         g_allParams.SD_CellIncThresh_OverTime[SD_ChangeTime] =
            g_allParams.SD_CellIncThresh_OverTime[g_allParams.Num_SD_ChangeTimes - 1];
      }

      //// case isolation
      for (int CI_ChangeTime = g_allParams.Num_CI_ChangeTimes; CI_ChangeTime < MAX_NUM_INTERVENTION_CHANGE_TIMES - 1;
           CI_ChangeTime++)
      {
         g_allParams.CI_SpatialAndPlaceEffects_OverTime[CI_ChangeTime] =
            g_allParams.CI_SpatialAndPlaceEffects_OverTime[g_allParams.Num_CI_ChangeTimes - 1];
         g_allParams.CI_HouseholdEffects_OverTime[CI_ChangeTime] =
            g_allParams.CI_HouseholdEffects_OverTime[g_allParams.Num_CI_ChangeTimes - 1];
         g_allParams.CI_Prop_OverTime[CI_ChangeTime] = g_allParams.CI_Prop_OverTime[g_allParams.Num_CI_ChangeTimes - 1];
         g_allParams.CI_CellIncThresh_OverTime[CI_ChangeTime] =
            g_allParams.CI_CellIncThresh_OverTime[g_allParams.Num_CI_ChangeTimes - 1];
      }

      //// household quarantine
      for (int HQ_ChangeTime = g_allParams.Num_HQ_ChangeTimes; HQ_ChangeTime < MAX_NUM_INTERVENTION_CHANGE_TIMES - 1;
           HQ_ChangeTime++)
      {
         g_allParams.HQ_SpatialEffects_OverTime[HQ_ChangeTime] =
            g_allParams.HQ_SpatialEffects_OverTime[g_allParams.Num_HQ_ChangeTimes - 1];
         g_allParams.HQ_HouseholdEffects_OverTime[HQ_ChangeTime] =
            g_allParams.HQ_HouseholdEffects_OverTime[g_allParams.Num_HQ_ChangeTimes - 1];
         for (int PlaceType = 0; PlaceType < g_allParams.PlaceTypeNum; PlaceType++)
            g_allParams.HQ_PlaceEffects_OverTime[HQ_ChangeTime][PlaceType] =
               g_allParams.HQ_PlaceEffects_OverTime[g_allParams.Num_HQ_ChangeTimes - 1][PlaceType];

         g_allParams.HQ_Individual_PropComply_OverTime[HQ_ChangeTime] =
            g_allParams.HQ_Individual_PropComply_OverTime[g_allParams.Num_HQ_ChangeTimes - 1];
         g_allParams.HQ_Household_PropComply_OverTime[HQ_ChangeTime] =
            g_allParams.HQ_Household_PropComply_OverTime[g_allParams.Num_HQ_ChangeTimes - 1];

         g_allParams.HQ_CellIncThresh_OverTime[HQ_ChangeTime] =
            g_allParams.HQ_CellIncThresh_OverTime[g_allParams.Num_HQ_ChangeTimes - 1];
      }

      //// place closure
      for (int PC_ChangeTime = g_allParams.Num_PC_ChangeTimes; PC_ChangeTime < MAX_NUM_INTERVENTION_CHANGE_TIMES - 1;
           PC_ChangeTime++)
      {
         g_allParams.PC_SpatialEffects_OverTime[PC_ChangeTime] =
            g_allParams.PC_SpatialEffects_OverTime[g_allParams.Num_PC_ChangeTimes - 1];
         g_allParams.PC_HouseholdEffects_OverTime[PC_ChangeTime] =
            g_allParams.PC_HouseholdEffects_OverTime[g_allParams.Num_PC_ChangeTimes - 1];
         for (int PlaceType = 0; PlaceType < g_allParams.PlaceTypeNum; PlaceType++)
         {
            g_allParams.PC_PlaceEffects_OverTime[PC_ChangeTime][PlaceType] =
               g_allParams.PC_PlaceEffects_OverTime[g_allParams.Num_PC_ChangeTimes - 1][PlaceType];
            g_allParams.PC_PropAttending_OverTime[PC_ChangeTime][PlaceType] =
               g_allParams.PC_PropAttending_OverTime[g_allParams.Num_PC_ChangeTimes - 1][PlaceType];
         }

         g_allParams.PC_IncThresh_OverTime[PC_ChangeTime] =
            g_allParams.PC_IncThresh_OverTime[g_allParams.Num_PC_ChangeTimes - 1];
         g_allParams.PC_FracIncThresh_OverTime[PC_ChangeTime] =
            g_allParams.PC_FracIncThresh_OverTime[g_allParams.Num_PC_ChangeTimes - 1];
         g_allParams.PC_CellIncThresh_OverTime[PC_ChangeTime] =
            g_allParams.PC_CellIncThresh_OverTime[g_allParams.Num_PC_ChangeTimes - 1];
      }

      //// digital contact tracing
      for (int DCT_ChangeTime = g_allParams.Num_DCT_ChangeTimes; DCT_ChangeTime < MAX_NUM_INTERVENTION_CHANGE_TIMES - 1;
           DCT_ChangeTime++)
      {
         g_allParams.DCT_SpatialAndPlaceEffects_OverTime[DCT_ChangeTime] =
            g_allParams.DCT_SpatialAndPlaceEffects_OverTime[g_allParams.Num_DCT_ChangeTimes - 1];
         g_allParams.DCT_HouseholdEffects_OverTime[DCT_ChangeTime] =
            g_allParams.DCT_HouseholdEffects_OverTime[g_allParams.Num_DCT_ChangeTimes - 1];
         g_allParams.DCT_Prop_OverTime[DCT_ChangeTime] =
            g_allParams.DCT_Prop_OverTime[g_allParams.Num_DCT_ChangeTimes - 1];
         g_allParams.DCT_MaxToTrace_OverTime[DCT_ChangeTime] =
            g_allParams.DCT_MaxToTrace_OverTime[g_allParams.Num_DCT_ChangeTimes - 1];
      }
   }

   if (g_allParams.DoPlaces)
   {
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Number of key workers randomly distributed in the population", "%i",
                              (void*)&(g_allParams.KeyWorkerPopNum), 1, 1, 0))
         g_allParams.KeyWorkerPopNum = 0;
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Number of key workers in different places by place type", "%i",
                              (void*)g_allParams.KeyWorkerPlaceNum, g_allParams.PlaceTypeNum, 1, 0))
      {
         for (int i = 0; i < NUM_PLACE_TYPES; i++)
         {
            g_allParams.KeyWorkerPlaceNum[i] = 0;
         }
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Proportion of staff who are key workers per chosen place by place type", "%lf",
                              (void*)g_allParams.KeyWorkerPropInKeyPlaces, g_allParams.PlaceTypeNum, 1, 0))
      {
         for (int i = 0; i < NUM_PLACE_TYPES; i++)
         {
            g_allParams.KeyWorkerPropInKeyPlaces[i] = 1.0;
         }
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Trigger incidence per cell for key worker prophylaxis",
                              "%i", (void*)&(g_allParams.KeyWorkerProphCellIncThresh), 1, 1, 0))
         g_allParams.KeyWorkerProphCellIncThresh = 1000000000;

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Key worker prophylaxis start time", "%lf",
                              (void*)&(g_allParams.KeyWorkerProphTimeStartBase), 1, 1, 0))
         g_allParams.KeyWorkerProphTimeStartBase = USHRT_MAX / g_allParams.TimeStepsPerDay;

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Duration of key worker prophylaxis", "%lf",
                              (void*)&(g_allParams.KeyWorkerProphDuration), 1, 1, 0))
         g_allParams.KeyWorkerProphDuration = 0;

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                              "Time interval from start of key worker prophylaxis before policy restarted", "%lf",
                              (void*)&(g_allParams.KeyWorkerProphRenewalDuration), 1, 1, 0))
         g_allParams.KeyWorkerProphRenewalDuration = g_allParams.KeyWorkerProphDuration;

      if (g_allParams.DoHouseholds)
      {
         if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat,
                                 "Proportion of key workers whose households are also treated as key workers", "%lf",
                                 (void*)&(g_allParams.KeyWorkerHouseProp), 1, 1, 0))
            g_allParams.KeyWorkerHouseProp = 0;
      }

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Minimum radius for key worker prophylaxis", "%lf",
                              (void*)&(g_allParams.KeyWorkerProphRadius), 1, 1, 0))
         g_allParams.KeyWorkerProphRadius = 0;
   }
   else
   {
      g_allParams.KeyWorkerPopNum             = 0;
      g_allParams.KeyWorkerProphTimeStartBase = 1e10;
   }

   // Added this to parameter list so that recording infection events (and the number to record) can easily be turned
   // off and on: ggilani - 10/10/2014
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Record infection events", "%i",
                           (void*)&(g_allParams.DoRecordInfEvents), 1, 1, 0))
      g_allParams.DoRecordInfEvents = 0;

   if (g_allParams.DoRecordInfEvents)
   {
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Max number of infection events to record", "%i",
                              (void*)&(g_allParams.MaxInfEvents), 1, 1, 0))
         g_allParams.MaxInfEvents = 1000;

      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Record infection events per run", "%i",
                              (void*)&(g_allParams.RecordInfEventsPerRun), 1, 1, 0))
         g_allParams.RecordInfEventsPerRun = 0;
   }
   else
   {
      g_allParams.MaxInfEvents = 0;
   }

   // Include a limit to the number of infections to simulate, if this happens before time runs out
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Limit number of infections", "%i",
                           (void*)&(g_allParams.LimitNumInfections), 1, 1, 0))
      g_allParams.LimitNumInfections = 0;

   if (g_allParams.LimitNumInfections)
   {
      if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Max number of infections", "%i",
                              (void*)&(g_allParams.MaxNumInfections), 1, 1, 0))
         g_allParams.MaxNumInfections = 60000;
   }
   // Add origin-destination matrix parameter
   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Output origin destination matrix", "%i",
                           (void*)&(g_allParams.DoOriginDestinationMatrix), 1, 1, 0))
      g_allParams.DoOriginDestinationMatrix = 0;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Mean child age gap", "%i",
                           (void*)&(g_allParams.MeanChildAgeGap), 1, 1, 0))
      g_allParams.MeanChildAgeGap = 2;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Min adult age", "%i", (void*)&(g_allParams.MinAdultAge), 1,
                           1, 0))
      g_allParams.MinAdultAge = 19;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Max MF partner age gap", "%i",
                           (void*)&(g_allParams.MaxMFPartnerAgeGap), 1, 1, 0))
      g_allParams.MaxMFPartnerAgeGap = 5;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Max FM partner age gap", "%i",
                           (void*)&(g_allParams.MaxFMPartnerAgeGap), 1, 1, 0))
      g_allParams.MaxFMPartnerAgeGap = 5;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Min parent age gap", "%i",
                           (void*)&(g_allParams.MinParentAgeGap), 1, 1, 0))
      g_allParams.MinParentAgeGap = 19;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Max parent age gap", "%i",
                           (void*)&(g_allParams.MaxParentAgeGap), 1, 1, 0))
      g_allParams.MaxParentAgeGap = 44;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Max child age", "%i", (void*)&(g_allParams.MaxChildAge), 1,
                           1, 0))
      g_allParams.MaxChildAge = 20;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "One Child Two Pers Prob", "%lf",
                           (void*)&(g_allParams.OneChildTwoPersProb), 1, 1, 0))
      g_allParams.OneChildTwoPersProb = 0.08;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Two Child Three Pers Prob", "%lf",
                           (void*)&(g_allParams.TwoChildThreePersProb), 1, 1, 0))
      g_allParams.TwoChildThreePersProb = 0.11;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "One Pers House Prob Old", "%lf",
                           (void*)&(g_allParams.OnePersHouseProbOld), 1, 1, 0))
      g_allParams.OnePersHouseProbOld = 0.5;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Two Pers House Prob Old", "%lf",
                           (void*)&(g_allParams.TwoPersHouseProbOld), 1, 1, 0))
      g_allParams.TwoPersHouseProbOld = 0.5;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "One Pers House Prob Young", "%lf",
                           (void*)&(g_allParams.OnePersHouseProbYoung), 1, 1, 0))
      g_allParams.OnePersHouseProbYoung = 0.23;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Two Pers House Prob Young", "%lf",
                           (void*)&(g_allParams.TwoPersHouseProbYoung), 1, 1, 0))
      g_allParams.TwoPersHouseProbYoung = 0.23;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "One Child Prob Youngest Child Under Five", "%lf",
                           (void*)&(g_allParams.OneChildProbYoungestChildUnderFive), 1, 1, 0))
      g_allParams.OneChildProbYoungestChildUnderFive = 0.5;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Two Children Prob Youngest Under Five", "%lf",
                           (void*)&(g_allParams.TwoChildrenProbYoungestUnderFive), 1, 1, 0))
      g_allParams.TwoChildrenProbYoungestUnderFive = 0.0;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Prob Youngest Child Under Five", "%lf",
                           (void*)&(g_allParams.ProbYoungestChildUnderFive), 1, 1, 0))
      g_allParams.ProbYoungestChildUnderFive = 0;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Zero Child Three Pers Prob", "%lf",
                           (void*)&(g_allParams.ZeroChildThreePersProb), 1, 1, 0))
      g_allParams.ZeroChildThreePersProb = 0.25;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "One Child Four Pers Prob", "%lf",
                           (void*)&(g_allParams.OneChildFourPersProb), 1, 1, 0))
      g_allParams.OneChildFourPersProb = 0.2;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Young And Single Slope", "%lf",
                           (void*)&(g_allParams.YoungAndSingleSlope), 1, 1, 0))
      g_allParams.YoungAndSingleSlope = 0.7;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Young And Single", "%i",
                           (void*)&(g_allParams.YoungAndSingle), 1, 1, 0))
      g_allParams.YoungAndSingle = 36;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "No Child Pers Age", "%i",
                           (void*)&(g_allParams.NoChildPersAge), 1, 1, 0))
      g_allParams.NoChildPersAge = 44;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Old Pers Age", "%i", (void*)&(g_allParams.OldPersAge), 1,
                           1, 0))
      g_allParams.OldPersAge = 60;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Three Child Five Pers Prob", "%lf",
                           (void*)&(g_allParams.ThreeChildFivePersProb), 1, 1, 0))
      g_allParams.ThreeChildFivePersProb = 0.5;

   if (!GetInputParameter2(ParamFile_dat, PreParamFile_dat, "Older Gen Gap", "%i", (void*)&(g_allParams.OlderGenGap), 1,
                           1, 0))
      g_allParams.OlderGenGap = 19;

   // Close input files.
   fclose(ParamFile_dat);

   if (PreParamFile_dat != NULL)
      fclose(PreParamFile_dat);

   if (ParamFile_dat != AdminFile_dat && AdminFile_dat != NULL)
      fclose(AdminFile_dat);

   if (g_allParams.DoOneGen != 0)
      g_allParams.DoOneGen = 1;

   g_allParams.ColourPeriod          = 2000;
   g_allParams.MoveRestrRadius2      = g_allParams.MoveRestrRadius * g_allParams.MoveRestrRadius;
   g_allParams.SocDistRadius2        = g_allParams.SocDistRadius * g_allParams.SocDistRadius;
   g_allParams.VaccRadius2           = g_allParams.VaccRadius * g_allParams.VaccRadius;
   g_allParams.VaccMinRadius2        = g_allParams.VaccMinRadius * g_allParams.VaccMinRadius;
   g_allParams.TreatRadius2          = g_allParams.TreatRadius * g_allParams.TreatRadius;
   g_allParams.PlaceCloseRadius2     = g_allParams.PlaceCloseRadius * g_allParams.PlaceCloseRadius;
   g_allParams.KeyWorkerProphRadius2 = g_allParams.KeyWorkerProphRadius * g_allParams.KeyWorkerProphRadius;
   if (g_allParams.TreatRadius2 == 0)
      g_allParams.TreatRadius2 = -1;

   if (g_allParams.VaccRadius2 == 0)
      g_allParams.VaccRadius2 = -1;

   if (g_allParams.PlaceCloseRadius2 == 0)
      g_allParams.PlaceCloseRadius2 = -1;

   if (g_allParams.MoveRestrRadius2 == 0)
      g_allParams.MoveRestrRadius2 = -1;

   if (g_allParams.SocDistRadius2 == 0)
      g_allParams.SocDistRadius2 = -1;

   if (g_allParams.KeyWorkerProphRadius2 == 0)
      g_allParams.KeyWorkerProphRadius2 = -1;

   if (g_allParams.TreatCellIncThresh < 1)
      g_allParams.TreatCellIncThresh = 1;

   if (g_allParams.CaseIsolation_CellIncThresh < 1)
      g_allParams.CaseIsolation_CellIncThresh = 1;

   if (g_allParams.DigitalContactTracing_CellIncThresh < 1)
      g_allParams.DigitalContactTracing_CellIncThresh = 1;

   if (g_allParams.HHQuar_CellIncThresh < 1)
      g_allParams.HHQuar_CellIncThresh = 1;

   if (g_allParams.MoveRestrCellIncThresh < 1)
      g_allParams.MoveRestrCellIncThresh = 1;

   if (g_allParams.PlaceCloseCellIncThresh < 1)
      g_allParams.PlaceCloseCellIncThresh = 1;

   if (g_allParams.KeyWorkerProphCellIncThresh < 1)
      g_allParams.KeyWorkerProphCellIncThresh = 1;

   //// Make unsigned short versions of various intervention variables. And scaled them by number of timesteps per day
   g_allParams.usHQuarantineHouseDuration =
      ((unsigned short int)(g_allParams.HQuarantineHouseDuration * g_allParams.TimeStepsPerDay));
   g_allParams.usVaccTimeToEfficacy =
      ((unsigned short int)(g_allParams.VaccTimeToEfficacy * g_allParams.TimeStepsPerDay));
   g_allParams.usVaccTimeEfficacySwitch =
      ((unsigned short int)(g_allParams.VaccTimeEfficacySwitch * g_allParams.TimeStepsPerDay));
   g_allParams.usCaseIsolationDelay =
      ((unsigned short int)(g_allParams.CaseIsolationDelay * g_allParams.TimeStepsPerDay));
   g_allParams.usCaseIsolationDuration =
      ((unsigned short int)(g_allParams.CaseIsolationDuration * g_allParams.TimeStepsPerDay));
   g_allParams.usCaseAbsenteeismDuration =
      ((unsigned short int)(g_allParams.CaseAbsenteeismDuration * g_allParams.TimeStepsPerDay));
   g_allParams.usCaseAbsenteeismDelay =
      ((unsigned short int)(g_allParams.CaseAbsenteeismDelay * g_allParams.TimeStepsPerDay));
   if (g_allParams.DoUTM_coords)
   {
      for (int i = 0; i <= 1000; i++)
      {
         asin2sqx[i] = asin(sqrt(((double)(i)) / 1000));
         asin2sqx[i] = asin2sqx[i] * asin2sqx[i];
      }
      for (t = 0; t <= 360; t++)
      {
         sinx[(int)t] = sin(PI * t / 180);
         cosx[(int)t] = cos(PI * t / 180);
      }
   }
   fprintf(stderr, "Parameters read\n");
}

void ReadInterventions(char* IntFile)
{
   FILE* dat;
   double r, s, startt, stopt;
   int j, k, au, ni, f, nsr;
   char buf[65536], txt[65536];
   intervention CurInterv;

   fprintf(stderr, "Reading intervention file.\n");
   if (!(dat = fopen(IntFile, "rb")))
      ERR_CRITICAL("Unable to open intervention file\n");
   if (fscanf(dat, "%*[^<]") != 0)
   { // needs to be separate line because start of file
      ERR_CRITICAL("fscanf failed in ReadInterventions\n");
   }
   if (fscanf(dat, "<%[^>]", txt) != 1)
   {
      ERR_CRITICAL("fscanf failed in ReadInterventions\n");
   }
   if (strcmp(txt, "\?xml version=\"1.0\" encoding=\"ISO-8859-1\"\?") != 0)
      ERR_CRITICAL("Intervention file not XML.\n");
   if (fscanf(dat, "%*[^<]<%[^>]", txt) != 1)
   {
      ERR_CRITICAL("fscanf failed in ReadInterventions\n");
   }
   if (strcmp(txt, "InterventionSettings") != 0)
      ERR_CRITICAL("Intervention has no top level.\n");
   ni = 0;
   while (!feof(dat))
   {
      if (fscanf(dat, "%*[^<]<%[^>]", txt) != 1)
      {
         ERR_CRITICAL("fscanf failed in ReadInterventions\n");
      }
      if (strcmp(txt, "intervention") == 0)
      {
         ni++;
         if (fscanf(dat, "%*[^<]<%[^>]", txt) != 1)
         {
            ERR_CRITICAL("fscanf failed in ReadInterventions\n");
         }
         if (strcmp(txt, "parameters") != 0)
            ERR_CRITICAL("Incomplete intervention parameter specification in intervention file\n");
         if (!GetXMLNode(dat, "Type", "parameters", txt, 1))
            ERR_CRITICAL("Incomplete intervention parameter specification in intervention file\n");
         if (strcmp(txt, "Treatment") == 0)
            CurInterv.InterventionType = 0;
         else if (strcmp(txt, "Vaccination") == 0)
            CurInterv.InterventionType = 1;
         else if (strcmp(txt, "ITN") == 0)
            CurInterv.InterventionType = 2;
         else if (strcmp(txt, "IRS") == 0)
            CurInterv.InterventionType = 3;
         else if (strcmp(txt, "GM") == 0)
            CurInterv.InterventionType = 4;
         else if (strcmp(txt, "MSAT") == 0)
            CurInterv.InterventionType = 5;
         else
            sscanf(txt, "%i", &CurInterv.InterventionType);
         if (!GetXMLNode(dat, "AUThresh", "parameters", txt, 1))
            ERR_CRITICAL("Incomplete intervention parameter specification in intervention file\n");
         sscanf(txt, "%i", &CurInterv.DoAUThresh);
         if (!GetXMLNode(dat, "StartTime", "parameters", txt, 1))
            ERR_CRITICAL("Incomplete intervention parameter specification in intervention file\n");
         sscanf(txt, "%lf", &CurInterv.StartTime);
         startt = CurInterv.StartTime;
         if (!GetXMLNode(dat, "StopTime", "parameters", txt, 1))
            ERR_CRITICAL("Incomplete intervention parameter specification in intervention file\n");
         sscanf(txt, "%lf", &CurInterv.StopTime);
         stopt = CurInterv.StopTime;
         if (!GetXMLNode(dat, "MinDuration", "parameters", txt, 1))
            ERR_CRITICAL("Incomplete intervention parameter specification in intervention file\n");
         sscanf(txt, "%lf", &CurInterv.MinDuration);
         CurInterv.MinDuration *= DAYS_PER_YEAR;
         if (!GetXMLNode(dat, "RepeatInterval", "parameters", txt, 1))
            ERR_CRITICAL("Incomplete intervention parameter specification in intervention file\n");
         sscanf(txt, "%lf", &CurInterv.RepeatInterval);
         CurInterv.RepeatInterval *= DAYS_PER_YEAR;
         if (!GetXMLNode(dat, "MaxPrevAtStart", "parameters", txt, 1))
            ERR_CRITICAL("Incomplete intervention parameter specification in intervention file\n");
         sscanf(txt, "%lf", &CurInterv.StartThresholdHigh);
         if (!GetXMLNode(dat, "MinPrevAtStart", "parameters", txt, 1))
            ERR_CRITICAL("Incomplete intervention parameter specification in intervention file\n");
         sscanf(txt, "%lf", &CurInterv.StartThresholdLow);
         if (!GetXMLNode(dat, "MaxPrevAtStop", "parameters", txt, 1))
            ERR_CRITICAL("Incomplete intervention parameter specification in intervention file\n");
         sscanf(txt, "%lf", &CurInterv.StopThreshold);
         if (GetXMLNode(dat, "NoStartAfterMinDur", "parameters", txt, 1))
            sscanf(txt, "%i", &CurInterv.NoStartAfterMin);
         else
            CurInterv.NoStartAfterMin = 0;
         if (!GetXMLNode(dat, "Level", "parameters", txt, 1))
            ERR_CRITICAL("Incomplete intervention parameter specification in intervention file\n");
         sscanf(txt, "%lf", &CurInterv.Level);
         if (GetXMLNode(dat, "LevelCellVar", "parameters", txt, 1))
            sscanf(txt, "%lf", &CurInterv.LevelCellVar);
         else
            CurInterv.LevelCellVar = 0;
         if (GetXMLNode(dat, "LevelAUVar", "parameters", txt, 1))
            sscanf(txt, "%lf", &CurInterv.LevelAUVar);
         else
            CurInterv.LevelCellVar = 0;
         if (GetXMLNode(dat, "LevelCountryVar", "parameters", txt, 1))
            sscanf(txt, "%lf", &CurInterv.LevelCountryVar);
         else
            CurInterv.LevelCellVar = 0;
         if (GetXMLNode(dat, "LevelClustering", "parameters", txt, 1))
            sscanf(txt, "%lf", &CurInterv.LevelClustering);
         else
            CurInterv.LevelClustering = 0;
         if (GetXMLNode(dat, "ControlParam", "parameters", txt, 1))
            sscanf(txt, "%lf", &CurInterv.ControlParam);
         else
            CurInterv.ControlParam = 0;
         if (GetXMLNode(dat, "TimeOffset", "parameters", txt, 1))
            sscanf(txt, "%lf", &CurInterv.TimeOffset);
         else
            CurInterv.TimeOffset = 0;

         if (!GetXMLNode(dat, "MaxRounds", "parameters", txt, 1))
            ERR_CRITICAL("Incomplete intervention parameter specification in intervention file\n");
         sscanf(txt, "%u", &CurInterv.MaxRounds);
         if (!GetXMLNode(dat, "MaxResource", "parameters", txt, 1))
            ERR_CRITICAL("Incomplete intervention parameter specification in intervention file\n");
         sscanf(txt, "%u", &CurInterv.MaxResource);
         if (GetXMLNode(dat, "NumSequentialReplicas", "parameters", txt, 1))
            sscanf(txt, "%i", &nsr);
         else
            nsr = 0;
         do
         {
            if (fscanf(dat, "%*[^<]<%[^>]", txt) != 1)
            {
               ERR_CRITICAL("fscanf failed in ReadInterventions\n");
            }
         } while ((strcmp(txt, "/intervention") != 0) && (strcmp(txt, "/parameters") != 0) && (!feof(dat)));
         if (strcmp(txt, "/parameters") != 0)
            ERR_CRITICAL("Incomplete intervention parameter specification in intervention file\n");
         if (fscanf(dat, "%*[^<]<%[^>]", txt) != 1)
         {
            ERR_CRITICAL("fscanf failed in ReadInterventions\n");
         }
         if ((strcmp(txt, "adunits") != 0) && (strcmp(txt, "countries") != 0))
            ERR_CRITICAL("Incomplete adunits/countries specification in intervention file\n");
         if (strcmp(txt, "adunits") == 0)
         {
            while (GetXMLNode(dat, "A", "adunits", buf, 0))
            {
               sscanf(buf, "%s", txt);
               j = atoi(txt);
               if (j == 0)
               {
                  f  = 1;
                  au = -1;
                  do
                  {
                     au++;
                     f = strcmp(txt, AdUnits[au].ad_name);
                  } while ((f) && (au < g_allParams.NumAdunits));
                  if (!f)
                  {
                     r = fabs(CurInterv.Level) + (2.0 * ranf() - 1) * CurInterv.LevelAUVar;
                     if ((CurInterv.Level < 1) && (r > 1))
                        r = 1;
                     else if (r < 0)
                        r = 0;
                     for (k = 0; k <= nsr; k++)
                     {
                        AdUnits[au].InterventionList[AdUnits[au].NI]       = CurInterv;
                        AdUnits[au].InterventionList[AdUnits[au].NI].Level = r;
                        AdUnits[au].InterventionList[AdUnits[au].NI].StartTime =
                           startt + ((double)k) * (stopt - startt);
                        AdUnits[au].InterventionList[AdUnits[au].NI].StopTime = stopt + ((double)k) * (stopt - startt);
                        AdUnits[au].NI++;
                     }
                  }
               }
               else
               {
                  k  = (j % g_allParams.AdunitLevel1Mask) / g_allParams.AdunitLevel1Divisor;
                  au = g_allParams.AdunitLevel1Lookup[k];
                  if ((au >= 0)
                      && (AdUnits[au].id / g_allParams.AdunitLevel1Divisor == j / g_allParams.AdunitLevel1Divisor))
                  {
                     r = CurInterv.Level + (2.0 * ranf() - 1) * CurInterv.LevelAUVar;
                     if ((CurInterv.Level < 1) && (r > 1))
                        r = 1;
                     else if (r < 0)
                        r = 0;
                     for (k = 0; k <= nsr; k++)
                     {
                        AdUnits[au].InterventionList[AdUnits[au].NI]       = CurInterv;
                        AdUnits[au].InterventionList[AdUnits[au].NI].Level = r;
                        AdUnits[au].InterventionList[AdUnits[au].NI].StartTime =
                           startt + ((double)k) * (stopt - startt);
                        AdUnits[au].InterventionList[AdUnits[au].NI].StopTime = stopt + ((double)k) * (stopt - startt);
                        AdUnits[au].NI++;
                     }
                  }
               }
            }
         }
         else
         {
            while (GetXMLNode(dat, "C", "countries", buf, 0))
            {
               s = (2.0 * ranf() - 1) * CurInterv.LevelCountryVar;
               sscanf(buf, "%s", txt);
               j = atoi(txt);
               for (au = 0; au < g_allParams.NumAdunits; au++)
                  if (((j == 0) && (strcmp(txt, AdUnits[au].cnt_name) == 0)) || ((j > 0) && (j == AdUnits[au].cnt_id)))
                  {
                     r = CurInterv.Level + (2.0 * ranf() - 1) * CurInterv.LevelAUVar + s;
                     if ((CurInterv.Level < 1) && (r > 1))
                        r = 1;
                     else if (r < 0)
                        r = 0;
                     for (k = 0; k <= nsr; k++)
                     {
                        AdUnits[au].InterventionList[AdUnits[au].NI]       = CurInterv;
                        AdUnits[au].InterventionList[AdUnits[au].NI].Level = r;
                        AdUnits[au].InterventionList[AdUnits[au].NI].StartTime =
                           startt + ((double)k) * (stopt - startt);
                        AdUnits[au].InterventionList[AdUnits[au].NI].StopTime = stopt + ((double)k) * (stopt - startt);
                        AdUnits[au].NI++;
                     }
                  }
            }
         }
         if (fscanf(dat, "%*[^<]<%[^>]", txt) != 1)
         {
            ERR_CRITICAL("fscanf failed in ReadInterventions\n");
         }
         if (strcmp(txt, "/intervention") != 0)
            ERR_CRITICAL("Incorrect intervention specification in intervention file\n");
      }
   }
   if (strcmp(txt, "/InterventionSettings") != 0)
      ERR_CRITICAL("Intervention has no top level closure.\n");
   fprintf(stderr, "%i interventions read\n", ni);
   fclose(dat);
}

int GetXMLNode(FILE* dat, const char* NodeName, const char* ParentName, char* Value, int ResetFilePos)
{
   // ResetFilePos=1 leaves dat cursor in same position as when function was called. 0 leaves it at end of NodeName
   // closure GetXMLNode returns 1 if NodeName found, 0 otherwise. If NodeName not found, ParentName closure must be

   char buf[65536], CloseNode[2048], CloseParent[2048];
   int CurPos, ret;

   sprintf(CloseParent, "/%s", ParentName);
   CurPos = ftell(dat);
   do
   {
      if (fscanf(dat, "%*[^<]<%[^>]", buf) != 1)
      {
         ERR_CRITICAL("fscanf failed in GetXMLNode");
      }
   } while ((strcmp(buf, CloseParent) != 0) && (strcmp(buf, NodeName) != 0) && (!feof(dat)));
   if (strcmp(buf, CloseParent) == 0)
      ret = 0;
   else
   {
      if (strcmp(buf, NodeName) != 0)
         ERR_CRITICAL("Incomplete node specification in XML file\n");
      if (fscanf(dat, ">%[^<]", buf) != 1)
      {
         ERR_CRITICAL("fscanf failed in GetXMLNode");
      }
      if (strlen(buf) < 2048)
         strcpy(Value, buf);
      //		fprintf(stderr,"# %s=%s\n",NodeName,Value);
      if (fscanf(dat, "<%[^>]", buf) != 1)
      {
         ERR_CRITICAL("fscanf failed in GetXMLNode");
      }
      sprintf(CloseNode, "/%s", NodeName);
      if (strcmp(buf, CloseNode) != 0)
         ERR_CRITICAL("Incomplete node specification in XML file\n");
      ret = 1;
   }
   if (ResetFilePos)
      fseek(dat, CurPos, 0);
   return ret;
}

void ReadAirTravel(char* AirTravelFile)
{
   int i, j, k, l;
   float sc, t, t2;
   float* buf;
   double traf;
   char outname[1024];
   FILE* dat;

   fprintf(stderr, "Reading airport data...\nAirports with no connections = ");
   if (!(dat = fopen(AirTravelFile, "rb")))
      ERR_CRITICAL("Unable to open airport file\n");
   if (fscanf(dat, "%i %i", &g_allParams.Nairports, &g_allParams.Air_popscale) != 2)
   {
      ERR_CRITICAL("fscanf failed in void ReadAirTravel\n");
   }
   sc = (float)((double)g_allParams.populationSize / (double)g_allParams.Air_popscale);
   if (g_allParams.Nairports > MAX_AIRPORTS)
      ERR_CRITICAL("Too many airports\n");
   if (g_allParams.Nairports < 2)
      ERR_CRITICAL("Too few airports\n");
   if (!(buf = (float*)calloc(g_allParams.Nairports + 1, sizeof(float))))
      ERR_CRITICAL("Unable to allocate airport storage\n");
   if (!(Airports = (airport*)calloc(g_allParams.Nairports, sizeof(airport))))
      ERR_CRITICAL("Unable to allocate airport storage\n");
   for (i = 0; i < g_allParams.Nairports; i++)
   {
      if (fscanf(dat, "%f %f %lf", &(Airports[i].loc_x), &(Airports[i].loc_y), &traf) != 3)
      {
         ERR_CRITICAL("fscanf failed in void ReadAirTravel\n");
      }
      traf *= (g_allParams.AirportTrafficScale * sc);
      if ((Airports[i].loc_x < g_allParams.SpatialBoundingBox[0])
          || (Airports[i].loc_x > g_allParams.SpatialBoundingBox[2])
          || (Airports[i].loc_y < g_allParams.SpatialBoundingBox[1])
          || (Airports[i].loc_y > g_allParams.SpatialBoundingBox[3]))
      {
         Airports[i].loc_x = Airports[i].loc_y = -1;
         Airports[i].total_traffic             = 0;
      }
      else
      {
         // fprintf(stderr,"(%f\t%f) ",Airports[i].loc_x,Airports[i].loc_y);
         Airports[i].loc_x -= (float)g_allParams.SpatialBoundingBox[0];
         Airports[i].loc_y -= (float)g_allParams.SpatialBoundingBox[1];
         Airports[i].total_traffic = (float)traf;
      }
      t = 0;
      for (j = k = 0; j < g_allParams.Nairports; j++)
      {
         if (fscanf(dat, "%f", buf + j) != 1)
         {
            ERR_CRITICAL("fscanf failed in void ReadAirTravel\n");
         }
         if (buf[j] > 0)
         {
            k++;
            t += buf[j];
         }
      }
      Airports[i].num_connected = k;
      if (Airports[i].num_connected > 0)
      {
         if (!(Airports[i].prop_traffic = (float*)calloc(Airports[i].num_connected, sizeof(float))))
            ERR_CRITICAL("Unable to allocate airport storage\n");
         if (!(Airports[i].conn_airports =
                  (unsigned short int*)calloc(Airports[i].num_connected, sizeof(unsigned short int))))
            ERR_CRITICAL("Unable to allocate airport storage\n");
         for (j = k = 0; j < g_allParams.Nairports; j++)
            if (buf[j] > 0)
            {
               Airports[i].conn_airports[k] = j;
               Airports[i].prop_traffic[k]  = buf[j] / t;
               k++;
            }
      }
      else
      {
         if (Airports[i].total_traffic > 0)
            fprintf(stderr, "#%i# ", i);
         else
            fprintf(stderr, "%i ", i);
      }
   }
   fclose(dat);
   free(buf);
   fprintf(stderr, "\nAirport data read OK.\n");
   for (i = 0; i < g_allParams.Nairports; i++)
   {
      /*		fprintf(stderr,"(%f %i|",Airports[i].total_traffic,Airports[i].num_connected);
       */
      t = 0;
      k = 0;
      for (j = Airports[i].num_connected - 1; j >= 0; j--)
      {
         if ((Airports[i].prop_traffic[j] > 0) && (Airports[Airports[i].conn_airports[j]].total_traffic == 0))
         {
            t += Airports[i].prop_traffic[j];
            Airports[i].num_connected--;
            if (j < Airports[i].num_connected)
            {
               Airports[i].prop_traffic[j]  = Airports[i].prop_traffic[Airports[i].num_connected];
               Airports[i].conn_airports[j] = Airports[i].conn_airports[Airports[i].num_connected];
            }
            Airports[i].prop_traffic[Airports[i].num_connected]  = 0;
            Airports[i].conn_airports[Airports[i].num_connected] = 0;
         }
         else if (Airports[i].prop_traffic[j] > 0)
            k = 1;
      }
      /*		fprintf(stderr,"%f %i ",t,k);
       */
      t = 1.0f - t;
      if (k)
      {
         Airports[i].total_traffic *= t;
         t2 = 0;
         for (j = 0; j < Airports[i].num_connected; j++)
         {
            Airports[i].prop_traffic[j] = t2 + Airports[i].prop_traffic[j];
            t2                          = Airports[i].prop_traffic[j];
         }
         for (j = 0; j < Airports[i].num_connected; j++)
            Airports[i].prop_traffic[j] /= t2;
         /*			if((Airports[i].num_connected>0)&&(Airports[i].prop_traffic[Airports[i].num_connected-1]!=1))
                     fprintf(stderr,"<%f> ",Airports[i].prop_traffic[Airports[i].num_connected-1]);
         */
      }
      else
      {
         Airports[i].total_traffic = 0;
         Airports[i].num_connected = 0;
      }
      if (Airports[i].num_connected > 0)
      {
         for (j = k = 0; k < 128; k++)
         {
            t = (float)((double)k / 128);
            while (Airports[i].prop_traffic[j] < t)
               j++;
            Airports[i].Inv_prop_traffic[k] = j;
         }
         Airports[i].Inv_prop_traffic[128] = Airports[i].num_connected - 1;
      }
      /*		fprintf(stderr,"%f) ",Airports[i].total_traffic);
       */
   }
   fprintf(stderr, "Airport data clipped OK.\n");
   for (i = 0; i < MAX_DIST; i++)
      AirTravelDist[i] = 0;
   for (i = 0; i < g_allParams.Nairports; i++)
      if (Airports[i].total_traffic > 0)
      {
         for (j = 0; j < Airports[i].num_connected; j++)
         {
            k    = (int)Airports[i].conn_airports[j];
            traf = floor(sqrt(dist2_raw(Airports[i].loc_x, Airports[i].loc_y, Airports[k].loc_x, Airports[k].loc_y))
                         / OUTPUT_DIST_SCALE);
            l    = (int)traf;
            // fprintf(stderr,"%(%i) ",l);
            if (l < MAX_DIST)
               AirTravelDist[l] += Airports[i].total_traffic * Airports[i].prop_traffic[j];
         }
      }
   sprintf(outname, "%s.airdist.xls", OutFilePath);
   if (!(dat = fopen(outname, "wb")))
      ERR_CRITICAL("Unable to open air travel output file\n");
   fprintf(dat, "dist\tfreq\n");
   for (i = 0; i < MAX_DIST; i++)
      fprintf(dat, "%i\t%.10f\n", i, AirTravelDist[i]);
   fclose(dat);
}

void InitModel(int run) // passing run number so we can save run number in the infection event log: ggilani - 15/10/2014
{
   int i, j, k, l, m, tn, nim;
   int nsi[MAX_NUM_SEED_LOCATIONS];

   if (g_allParams.OutputBitmap)
   {
#ifdef WIN32_BM
      // if (g_allParams.OutputBitmap == 1)
      //{
      //	char buf[200];
      //	sprintf(buf, "%s.ge" DIRECTORY_SEPARATOR "%s.avi", OutFile, OutFile);
      //	avi = CreateAvi(buf, g_allParams.BitmapMovieFrame, NULL);
      //}
#endif
      for (unsigned p = 0; p < bmh->imagesize; p++)
      {
         bmInfected[p] = bmRecovered[p] = bmTreated[p] = 0;
      }
   }

   // TODO: Remove this, it's unused.
   // ns      = 0;

   State.S = g_allParams.populationSize;
   State.L = State.I = State.R = State.D = 0;
   State.cumI = State.cumR = State.cumC = State.cumFC = State.cumH = State.cumCT = State.cumCC = State.cumTC =
      State.cumD = State.cumDC = State.trigDC = State.DCT = State.cumDCT = State.cumInf_h = State.cumInf_n =
         State.cumInf_s = State.cumHQ = State.cumAC = State.cumAH = State.cumAA = State.cumACS = State.cumAPC =
            State.cumAPA = State.cumAPCS = 0;
   State.cumT = State.cumUT = State.cumTP = State.cumV = State.sumRad2 = State.maxRad2 = State.cumV_daily =
      State.cumVG  = 0; // added State.cumVG
   State.mvacc_cum = 0;
   if (g_allParams.DoSeverity)
   {
      State.Mild = State.ILI = State.SARI = State.Critical = State.CritRecov = 0;
      State.cumMild = State.cumILI = State.cumSARI = State.cumCritical = State.cumCritRecov = 0;
      State.cumDeath_ILI = State.cumDeath_SARI = State.cumDeath_Critical = 0;

      for (int AdminUnit = 0; AdminUnit <= g_allParams.NumAdunits; AdminUnit++)
      {
         State.Mild_adunit[AdminUnit] = State.ILI_adunit[AdminUnit] = State.SARI_adunit[AdminUnit] =
            State.Critical_adunit[AdminUnit] = State.CritRecov_adunit[AdminUnit] = State.cumMild_adunit[AdminUnit] =
               State.cumILI_adunit[AdminUnit] = State.cumSARI_adunit[AdminUnit] = State.cumCritical_adunit[AdminUnit] =
                  State.cumCritRecov_adunit[AdminUnit]                          = State.cumDeath_ILI_adunit[AdminUnit] =
                     State.cumDeath_SARI_adunit[AdminUnit] = State.cumDeath_Critical_adunit[AdminUnit] =
                        State.cumD_adunit[AdminUnit]       = 0;
      }
   }

   for (i = 0; i < NUM_AGE_GROUPS; i++)
      State.cumCa[i] = State.cumIa[i] = State.cumDa[i] = 0;
   for (i = 0; i < 2; i++)
      State.cumC_keyworker[i] = State.cumI_keyworker[i] = State.cumT_keyworker[i] = 0;
   for (i = 0; i < NUM_PLACE_TYPES; i++)
      State.NumPlacesClosed[i] = 0;
   for (i = 0; i < INFECT_TYPE_MASK; i++)
      State.cumItype[i] = 0;
   // initialise cumulative case counts per country to zero: ggilani 12/11/14
   for (i = 0; i < MAX_COUNTRIES; i++)
      State.cumC_country[i] = 0;
   if (g_allParams.DoAdUnits)
      for (i = 0; i <= g_allParams.NumAdunits; i++)
      {
         State.cumI_adunit[i] = State.cumC_adunit[i] = State.cumD_adunit[i] = State.cumT_adunit[i] =
            State.cumH_adunit[i] = State.cumDC_adunit[i] = State.cumCT_adunit[i] = State.cumCC_adunit[i] =
               State.trigDC_adunit[i] = State.DCT_adunit[i] = State.cumDCT_adunit[i] =
                  0; // added hospitalisation, added detected cases, contact tracing per adunit, cases who are contacts:
                     // ggilani 03/02/15, 15/06/17
         AdUnits[i].place_close_trig                  = 0;
         AdUnits[i].CaseIsolationTimeStart            = AdUnits[i].HQuarantineTimeStart =
            AdUnits[i].DigitalContactTracingTimeStart = AdUnits[i].SocialDistanceTimeStart =
               AdUnits[i].PlaceCloseTimeStart         = 1e10;
         AdUnits[i].ndct                              = 0; // noone being digitally contact traced at beginning of run
      }

   // update state variables for storing contact distribution
   for (i = 0; i < MAX_CONTACTS + 1; i++)
      State.contact_dist[i] = 0;

   for (j = 0; j < MAX_NUM_THREADS; j++)
   {
      StateT[j].L = StateT[j].I = StateT[j].R = StateT[j].D = 0;
      StateT[j].cumI = StateT[j].cumR = StateT[j].cumC = StateT[j].cumFC = StateT[j].cumH = StateT[j].cumCT =
         StateT[j].cumCC = StateT[j].DCT = StateT[j].cumDCT = StateT[j].cumTC = StateT[j].cumD = StateT[j].cumDC =
            StateT[j].cumInf_h = StateT[j].cumInf_n = StateT[j].cumInf_s = StateT[j].cumHQ = StateT[j].cumAC =
               StateT[j].cumACS = StateT[j].cumAH = StateT[j].cumAA = StateT[j].cumAPC = StateT[j].cumAPA =
                  StateT[j].cumAPCS                                                    = 0;
      StateT[j].cumT = StateT[j].cumUT = StateT[j].cumTP = StateT[j].cumV = StateT[j].sumRad2 = StateT[j].maxRad2 =
         StateT[j].cumV_daily                                                                 = 0;
      for (i = 0; i < NUM_AGE_GROUPS; i++)
         StateT[j].cumCa[i] = StateT[j].cumIa[i] = StateT[j].cumDa[i] = 0;
      for (i = 0; i < 2; i++)
         StateT[j].cumC_keyworker[i] = StateT[j].cumI_keyworker[i] = StateT[j].cumT_keyworker[i] = 0;
      for (i = 0; i < NUM_PLACE_TYPES; i++)
         StateT[j].NumPlacesClosed[i] = 0;
      for (i = 0; i < INFECT_TYPE_MASK; i++)
         StateT[j].cumItype[i] = 0;
      // initialise cumulative case counts per country per thread to zero: ggilani 12/11/14
      for (i = 0; i < MAX_COUNTRIES; i++)
         StateT[j].cumC_country[i] = 0;
      if (g_allParams.DoAdUnits)
         for (i = 0; i <= g_allParams.NumAdunits; i++)
            StateT[j].cumI_adunit[i] = StateT[j].cumC_adunit[i] = StateT[j].cumD_adunit[i] = StateT[j].cumT_adunit[i] =
               StateT[j].cumH_adunit[i] = StateT[j].cumDC_adunit[i] = StateT[j].cumCT_adunit[i] =
                  StateT[j].cumCC_adunit[i] = StateT[j].nct_queue[i] = StateT[j].cumDCT_adunit[i] =
                     StateT[j].DCT_adunit[i]                         = StateT[j].ndct_queue[i] =
                        0; // added hospitalisation, detected cases, contact tracing per adunit, cases who are contacts:
                           // ggilani 03/02/15, 15/06/17

      if (g_allParams.DoSeverity)
      {
         StateT[j].Mild = StateT[j].ILI = StateT[j].SARI = StateT[j].Critical = StateT[j].CritRecov = 0;
         StateT[j].cumMild = StateT[j].cumILI = StateT[j].cumSARI = StateT[j].cumCritical = StateT[j].cumCritRecov = 0;
         StateT[j].cumDeath_ILI = StateT[j].cumDeath_SARI = StateT[j].cumDeath_Critical = 0;

         for (int AdminUnit = 0; AdminUnit <= g_allParams.NumAdunits; AdminUnit++)
         {
            StateT[j].Mild_adunit[AdminUnit] = StateT[j].ILI_adunit[AdminUnit] = StateT[j].SARI_adunit[AdminUnit] =
               StateT[j].Critical_adunit[AdminUnit]                            = StateT[j].CritRecov_adunit[AdminUnit] =
                  StateT[j].cumMild_adunit[AdminUnit]                          = StateT[j].cumILI_adunit[AdminUnit] =
                     StateT[j].cumSARI_adunit[AdminUnit]             = StateT[j].cumCritical_adunit[AdminUnit] =
                        StateT[j].cumCritRecov_adunit[AdminUnit]     = StateT[j].cumDeath_ILI_adunit[AdminUnit] =
                           StateT[j].cumDeath_SARI_adunit[AdminUnit] = StateT[j].cumDeath_Critical_adunit[AdminUnit] =
                              StateT[j].cumD_adunit[AdminUnit]       = 0;
         }
      }
      // resetting thread specific parameters for storing contact distribution
      for (i = 0; i < MAX_CONTACTS + 1; i++)
         StateT[j].contact_dist[i] = 0;
   }
   nim = 0;

#pragma omp parallel for private(tn, k) schedule(static, 1)
   for (tn = 0; tn < g_allParams.NumThreads; tn++)
      for (k = tn; k < g_allParams.populationSize; k += g_allParams.NumThreads)
      {
         Hosts[k].absent_start_time = USHRT_MAX - 1;
         Hosts[k].absent_stop_time  = 0;
         if (g_allParams.DoAirports)
            Hosts[k].PlaceLinks[g_allParams.HotelPlaceType] = -1;
         Hosts[k].vacc_start_time = Hosts[k].treat_start_time = Hosts[k].quar_start_time =
            Hosts[k].isolation_start_time = Hosts[k].absent_start_time = Hosts[k].dct_start_time =
               Hosts[k].dct_trigger_time                               = USHRT_MAX - 1;
         Hosts[k].treat_stop_time = Hosts[k].absent_stop_time = Hosts[k].dct_end_time = 0;
         Hosts[k].quar_comply                                                         = 2;
         Hosts[k].susc = (g_allParams.DoPartialImmunity) ? (1.0 - g_allParams.InitialImmunity[HOST_AGE_GROUP(k)]) : 1.0;
         Hosts[k].to_die     = 0;
         Hosts[k].Travelling = 0;
         Hosts[k].detected = 0; // set detected to zero initially: ggilani - 19/02/15
         Hosts[k].detected_time        = 0;
         Hosts[k].digitalContactTraced = 0;
         Hosts[k].inf                  = InfStat_Susceptible;
         Hosts[k].num_treats           = 0;
         Hosts[k].latent_time          = Hosts[k].recovery_or_death_time =
            0; // also set hospitalisation time to zero: ggilani 28/10/2014
         Hosts[k].infector       = -1;
         Hosts[k].infect_type    = 0;
         Hosts[k].index_case_dct = 0;
         Hosts[k].ProbAbsent     = (float)ranf_mt(tn);
         Hosts[k].ProbCare       = (float)ranf_mt(tn);
         if (g_allParams.DoSeverity)
         {
            Hosts[k].SARI_time =
               USHRT_MAX
               - 1; //// think better to set to initialize to maximum possible value, but keep this way for now.
            Hosts[k].Critical_time               = USHRT_MAX - 1;
            Hosts[k].RecoveringFromCritical_time = USHRT_MAX - 1;
            Hosts[k].Severity_Current            = Severity_Asymptomatic;
            Hosts[k].Severity_Final              = Severity_Asymptomatic;
            Hosts[k].inf                         = InfStat_Susceptible;
         }
      }

#pragma omp parallel for private(i, j, k, l, m, tn) reduction(+ : nim) schedule(static, 1)
   for (tn = 0; tn < g_allParams.NumThreads; tn++)
   {
      for (i = tn; i < g_allParams.cellCount; i += g_allParams.NumThreads)
      {
         if ((Cells[i].tot_treat != 0) || (Cells[i].tot_vacc != 0) || (Cells[i].S != Cells[i].n) || (Cells[i].D > 0)
             || (Cells[i].R > 0))
         {
            for (j = 0; j < Cells[i].n; j++)
            {
               k                       = Cells[i].members[j];
               Cells[i].susceptible[j] = k; // added this in here instead
               Hosts[k].listpos        = j;
            }
            Cells[i].S = Cells[i].n;
            Cells[i].L = Cells[i].I = Cells[i].R = Cells[i].cumTC = Cells[i].D = 0;
            Cells[i].infected = Cells[i].latent = Cells[i].susceptible + Cells[i].S;
            Cells[i].tot_treat = Cells[i].tot_vacc = 0;
            for (l = 0; l < MAX_INTERVENTION_TYPES; l++)
               Cells[i].CurInterv[l] = -1;

            // Next loop needs to count down for DoImmune host list reordering to work
            if (!g_allParams.DoPartialImmunity)
               for (j = Cells[i].n - 1; j >= 0; j--)
               {
                  k = Cells[i].members[j];
                  if (g_allParams.DoWholeHouseholdImmunity)
                  {
                     // note that this breaks determinism of runs if executed due to reordering of Cell members list
                     // each realisation
                     if (g_allParams.InitialImmunity[0] != 0)
                     {
                        if (Households[Hosts[k].hh].FirstPerson == k)
                        {
                           if ((g_allParams.InitialImmunity[0] == 1) || (ranf_mt(tn) < g_allParams.InitialImmunity[0]))
                           {
                              nim += Households[Hosts[k].hh].nh;
                              for (m = Households[Hosts[k].hh].nh - 1; m >= 0; m--)
                                 DoImmune(k + m);
                           }
                        }
                     }
                  }
                  else
                  {
                     m = HOST_AGE_GROUP(k);
                     if ((g_allParams.InitialImmunity[m] == 1)
                         || ((g_allParams.InitialImmunity[m] > 0) && (ranf_mt(tn) < g_allParams.InitialImmunity[m])))
                     {
                        DoImmune(k);
                        nim += 1;
                     }
                  }
               }
         }
      }
   }

#pragma omp parallel for private(i, j, k, l) schedule(static, 500)
   for (l = 0; l < g_allParams.NMCP; l++)
   {
      i                         = (int)(McellLookup[l] - Mcells);
      Mcells[i].vacc_start_time = Mcells[i].treat_start_time = USHRT_MAX - 1;
      Mcells[i].treat_end_time                               = 0;
      Mcells[i].treat_trig = Mcells[i].vacc_trig = Mcells[i].vacc = Mcells[i].treat = 0;
      Mcells[i].place_trig = Mcells[i].move_trig = Mcells[i].socdist_trig = Mcells[i].keyworkerproph_trig =
         Mcells[i].placeclose = Mcells[i].moverest = Mcells[i].socdist = Mcells[i].keyworkerproph = 0;
      Mcells[i].move_start_time                                                                   = USHRT_MAX - 1;
      Mcells[i].place_end_time = Mcells[i].move_end_time = Mcells[i].socdist_end_time =
         Mcells[i].keyworkerproph_end_time               = 0;
   }
   if (g_allParams.DoPlaces)
#pragma omp parallel for private(m, l) schedule(static, 1)
      for (m = 0; m < g_allParams.PlaceTypeNum; m++)
      {
         for (l = 0; l < g_allParams.Nplace[m]; l++)
         {
            Places[m][l].close_start_time = USHRT_MAX - 1;
            Places[m][l].treat = Places[m][l].control_trig = 0;
            Places[m][l].treat_end_time = Places[m][l].close_end_time = 0;
            Places[m][l].ProbClose                                    = (float)ranf_mt(m);
            if (g_allParams.AbsenteeismPlaceClosure)
            {
               Places[m][l].AbsentLastUpdateTime = 0;
               for (int i2 = 0; i2 < g_allParams.MaxAbsentTime; i2++)
                  Places[m][l].Absent[i2] = 0;
            }
         }
      }

   //// **** //// **** //// **** Initialize Current effects
   //// **** soc dist
   g_allParams.SocDistDurationCurrent        = g_allParams.SocDistDuration;
   g_allParams.SocDistSpatialEffectCurrent   = g_allParams.SD_SpatialEffects_OverTime[0];   //// spatial
   g_allParams.SocDistHouseholdEffectCurrent = g_allParams.SD_HouseholdEffects_OverTime[0]; //// household
   for (int PlaceType = 0; PlaceType < g_allParams.PlaceTypeNum; PlaceType++)
      g_allParams.SocDistPlaceEffectCurrent[PlaceType] = g_allParams.SD_PlaceEffects_OverTime[0][PlaceType]; //// place
   g_allParams.SocDistCellIncThresh = g_allParams.SD_CellIncThresh_OverTime[0]; //// cell incidence threshold

   //// **** enhanced soc dist
   g_allParams.EnhancedSocDistSpatialEffectCurrent = g_allParams.Enhanced_SD_SpatialEffects_OverTime[0]; //// spatial
   g_allParams.EnhancedSocDistHouseholdEffectCurrent =
      g_allParams.Enhanced_SD_HouseholdEffects_OverTime[0]; //// household
   for (int PlaceType = 0; PlaceType < g_allParams.PlaceTypeNum; PlaceType++)
      g_allParams.EnhancedSocDistPlaceEffectCurrent[PlaceType] =
         g_allParams.Enhanced_SD_PlaceEffects_OverTime[0][PlaceType]; //// place

   //// **** case isolation
   g_allParams.CaseIsolationEffectiveness      = g_allParams.CI_SpatialAndPlaceEffects_OverTime[0]; //// spatial / place
   g_allParams.CaseIsolationHouseEffectiveness = g_allParams.CI_HouseholdEffects_OverTime[0];       //// household
   g_allParams.CaseIsolationProp               = g_allParams.CI_Prop_OverTime[0];                   //// compliance
   g_allParams.CaseIsolation_CellIncThresh     = g_allParams.CI_CellIncThresh_OverTime[0]; //// cell incidence threshold

   //// **** household quarantine
   g_allParams.HQuarantineSpatialEffect = g_allParams.HQ_SpatialEffects_OverTime[0];   //// spatial
   g_allParams.HQuarantineHouseEffect   = g_allParams.HQ_HouseholdEffects_OverTime[0]; //// household
   for (int PlaceType = 0; PlaceType < g_allParams.PlaceTypeNum; PlaceType++)
      g_allParams.HQuarantinePlaceEffect[PlaceType] = g_allParams.HQ_PlaceEffects_OverTime[0][PlaceType]; //// place
   g_allParams.HQuarantinePropIndivCompliant =
      g_allParams.HQ_Individual_PropComply_OverTime[0]; //// individual compliance
   g_allParams.HQuarantinePropHouseCompliant =
      g_allParams.HQ_Household_PropComply_OverTime[0];                          //// household compliance
   g_allParams.HHQuar_CellIncThresh = g_allParams.HQ_CellIncThresh_OverTime[0]; //// cell incidence threshold

   //// **** place closure
   g_allParams.PlaceCloseSpatialRelContact   = g_allParams.PC_SpatialEffects_OverTime[0];   //// spatial
   g_allParams.PlaceCloseHouseholdRelContact = g_allParams.PC_HouseholdEffects_OverTime[0]; //// household
   for (int PlaceType = 0; PlaceType < g_allParams.PlaceTypeNum; PlaceType++)
   {
      g_allParams.PlaceCloseEffect[PlaceType]        = g_allParams.PC_PlaceEffects_OverTime[0][PlaceType]; //// place
      g_allParams.PlaceClosePropAttending[PlaceType] = g_allParams.PC_PropAttending_OverTime[0][PlaceType];
   }
   g_allParams.PlaceCloseIncTrig1       = g_allParams.PC_IncThresh_OverTime[0];     //// global incidence threshold
   g_allParams.PlaceCloseFracIncTrig    = g_allParams.PC_FracIncThresh_OverTime[0]; //// fractional incidence threshold
   g_allParams.PlaceCloseCellIncThresh1 = g_allParams.PC_CellIncThresh_OverTime[0]; //// cell incidence threshold
   g_allParams.PlaceCloseDurationBase   = g_allParams.PC_Durs_OverTime[0];          //// duration of place closure

   //// **** digital contact tracing
   g_allParams.DCTCaseIsolationEffectiveness = g_allParams.DCT_SpatialAndPlaceEffects_OverTime[0]; //// spatial / place
   g_allParams.DCTCaseIsolationHouseEffectiveness = g_allParams.DCT_HouseholdEffects_OverTime[0];  //// household
   g_allParams.ProportionDigitalContactsIsolate   = g_allParams.DCT_Prop_OverTime[0];              //// compliance
   g_allParams.MaxDigitalContactsToTrace          = g_allParams.DCT_MaxToTrace_OverTime[0];

   for (i = 0; i < MAX_NUM_THREADS; i++)
   {
      for (j = 0; j < MAX_NUM_THREADS; j++)
         StateT[i].n_queue[j] = 0;
      for (j = 0; j < g_allParams.PlaceTypeNum; j++)
         StateT[i].np_queue[j] = 0;
   }
   if (DoInitUpdateProbs)
   {
      UpdateProbs(0);
      DoInitUpdateProbs = 0;
   }
   // initialise event log to zero at the beginning of every run: ggilani - 10/10/2014. UPDATE: 15/10/14 - we are now
   // going to store all events from all realisations in one file
   if ((g_allParams.DoRecordInfEvents) && (g_allParams.RecordInfEventsPerRun))
   {
      *nEvents = 0;
      for (i = 0; i < g_allParams.MaxInfEvents; i++)
      {
         InfEventLog[i].t = InfEventLog[i].infectee_x = InfEventLog[i].infectee_y = InfEventLog[i].t_infector = 0.0;
         InfEventLog[i].infectee_ind = InfEventLog[i].infector_ind = 0;
         InfEventLog[i].infectee_adunit = InfEventLog[i].listpos = InfEventLog[i].infectee_cell =
            InfEventLog[i].infector_cell = InfEventLog[i].thread = 0;
      }
   }

   for (i = 0; i < g_allParams.NumSeedLocations; i++)
      nsi[i] = (int)(((double)g_allParams.NumInitialInfections[i]) * g_allParams.InitialInfectionsAdminUnitWeight[i]
                        * g_allParams.SeedingScaling
                     + 0.5);
   SeedInfection(0, nsi, 0, run);
   g_allParams.ControlPropCasesId = g_allParams.PreAlertControlPropCasesId;
   g_allParams.TreatTimeStart     = 1e10;

   g_allParams.VaccTimeStart          = 1e10;
   g_allParams.MoveRestrTimeStart     = 1e10;
   g_allParams.PlaceCloseTimeStart    = 1e10;
   g_allParams.PlaceCloseTimeStart2   = 2e10;
   g_allParams.SocDistTimeStart       = 1e10;
   g_allParams.AirportCloseTimeStart  = 1e10;
   g_allParams.CaseIsolationTimeStart = 1e10;
   // g_allParams.DigitalContactTracingTimeStart = 1e10;
   g_allParams.HQuarantineTimeStart        = 1e10;
   g_allParams.KeyWorkerProphTimeStart     = 1e10;
   g_allParams.TreatMaxCourses             = g_allParams.TreatMaxCoursesBase;
   g_allParams.VaccMaxCourses              = g_allParams.VaccMaxCoursesBase;
   g_allParams.PlaceCloseDuration          = g_allParams.PlaceCloseDurationBase; //// duration of place closure
   g_allParams.PlaceCloseIncTrig           = g_allParams.PlaceCloseIncTrig1;
   g_allParams.PlaceCloseTimeStartPrevious = 1e10;
   g_allParams.PlaceCloseCellIncThresh     = g_allParams.PlaceCloseCellIncThresh1;
   g_allParams.ResetSeedsFlag              = 0; // added this to allow resetting seeds part way through run: ggilani 27/11/2019
   if (!g_allParams.StopCalibration)
      g_allParams.PreControlClusterIdTime = 0;

   fprintf(stderr, "Finished InitModel.\n");
}

void SeedInfection(double t, int* nsi, int rf, int run) // adding run number to pass it to event log
{
   /* *nsi is an array of the number of seeding infections by location (I think). During runtime, usually just a single
    * int (given by a poisson distribution)*/
   /*rf set to 0 when initializing model, otherwise set to 1 during runtime. */

   int i /*seed location index*/;
   int j /*microcell number*/;
   int k, l /*k,l are grid coords at first, then l changed to be person within Microcell j, then k changed to be index
               of new infection*/
      ;
   int m = 0 /*guard against too many infections and infinite loop*/;
   int f /*range = {0, 1000}*/;
   int n /*number of seed locations?*/;

   n = ((rf == 0) ? g_allParams.NumSeedLocations : 1);
   for (i = 0; i < n; i++)
   {
      if ((!g_allParams.DoRandomInitialInfectionLoc)
          || ((g_allParams.DoAllInitialInfectioninSameLoc)
              && (rf))) //// either non-random locations, doing all initial infections in same location, and not
                        ///initializing.
      {
         k = (int)(g_allParams.LocationInitialInfection[i][0] / g_allParams.mcwidth);
         l = (int)(g_allParams.LocationInitialInfection[i][1] / g_allParams.mcheight);
         j = k * g_allParams.nmch + l;
         m = 0;
         for (k = 0; (k < nsi[i]) && (m < 10000); k++)
         {
            l = Mcells[j].members[(
               int)(ranf() * ((double)Mcells[j].n))]; //// randomly choose member of microcell j. Name this member l
            if (Hosts[l].inf == InfStat_Susceptible)  //// If Host l is uninfected.
            {
               if (CalcPersonSusc(l, 0, 0, 0) > 0)
               {
                  // only reset the initial location if rf==0, i.e. when initial seeds are being set, not when imported
                  // cases are being set
                  if (rf == 0)
                  {
                     g_allParams.LocationInitialInfection[i][0] = Households[Hosts[l].hh].loc_x;
                     g_allParams.LocationInitialInfection[i][1] = Households[Hosts[l].hh].loc_y;
                  }
                  Hosts[l].infector    = -2;
                  Hosts[l].infect_type = INFECT_TYPE_MASK - 1;
                  DoInfect(l, t, 0, run); ///// guessing this updates a number of things about person l at time t in
                                          ///thread 0 for this run.
                  m = 0;
               }
            }
            else
            {
               k--;
               m++;
            } //// think k-- means if person l chosen is already infected, go again. The m < 10000 is a guard against a)
              ///too many infections; b) an infinite loop if no more uninfected people left.
         }
      }
      else if (g_allParams.DoAllInitialInfectioninSameLoc)
      {
         f = 0;
         do
         {
            m = 0;
            do
            {
               l = (int)(ranf() * ((double)g_allParams.populationSize));
               j = Hosts[l].mcell;
               // fprintf(stderr,"%i ",AdUnits[Mcells[j].adunit].id);
            } while (
               (Mcells[j].n < nsi[i]) || (Mcells[j].n > g_allParams.MaxPopDensForInitialInfection)
               || (Mcells[j].n < g_allParams.MinPopDensForInitialInfection)
               || ((g_allParams.InitialInfectionsAdminUnit[i] > 0)
                   && ((AdUnits[Mcells[j].adunit].id % g_allParams.AdunitLevel1Mask) / g_allParams.AdunitLevel1Divisor
                       != (g_allParams.InitialInfectionsAdminUnit[i] % g_allParams.AdunitLevel1Mask)
                             / g_allParams.AdunitLevel1Divisor)));
            for (k = 0; (k < nsi[i]) && (m < 10000); k++)
            {
               l = Mcells[j].members[(int)(ranf() * ((double)Mcells[j].n))];
               if (Hosts[l].inf == InfStat_Susceptible)
               {
                  if (CalcPersonSusc(l, 0, 0, 0) > 0)
                  {
                     g_allParams.LocationInitialInfection[i][0] = Households[Hosts[l].hh].loc_x;
                     g_allParams.LocationInitialInfection[i][1] = Households[Hosts[l].hh].loc_y;
                     Hosts[l].infector                          = -2;
                     Hosts[l].infect_type                       = INFECT_TYPE_MASK - 1;
                     DoInfect(l, t, 0, run);
                     m = 0;
                  }
               }
               else
               {
                  k--;
                  m++;
               }
            }
            if (m)
               f++;
            else
               f = 0;
         } while ((f > 0) && (f < 1000));
      }
      else
      {
         m = 0;
         for (k = 0; (k < nsi[i]) && (m < 10000); k++)
         {
            do
            {
               l = (int)(ranf() * ((double)g_allParams.populationSize));
               j = Hosts[l].mcell;
               // fprintf(stderr,"@@ %i %i ",AdUnits[Mcells[j].adunit].id, (int)(AdUnits[Mcells[j].adunit].id /
               // g_allParams.CountryDivisor));
            } while (
               (Mcells[j].n == 0) || (Mcells[j].n > g_allParams.MaxPopDensForInitialInfection)
               || (Mcells[j].n < g_allParams.MinPopDensForInitialInfection)
               || ((g_allParams.InitialInfectionsAdminUnit[i] > 0)
                   && ((AdUnits[Mcells[j].adunit].id % g_allParams.AdunitLevel1Mask) / g_allParams.AdunitLevel1Divisor
                       != (g_allParams.InitialInfectionsAdminUnit[i] % g_allParams.AdunitLevel1Mask)
                             / g_allParams.AdunitLevel1Divisor)));
            l = Mcells[j].members[(int)(ranf() * ((double)Mcells[j].n))];
            if (Hosts[l].inf == InfStat_Susceptible)
            {
               if (CalcPersonSusc(l, 0, 0, 0) > 0)
               {
                  g_allParams.LocationInitialInfection[i][0] = Households[Hosts[l].hh].loc_x;
                  g_allParams.LocationInitialInfection[i][1] = Households[Hosts[l].hh].loc_y;
                  Hosts[l].infector                          = -2;
                  Hosts[l].infect_type                       = INFECT_TYPE_MASK - 1;
                  DoInfect(l, t, 0, run);
                  m = 0;
               }
               else
               {
                  k--;
                  m++;
               }
            }
            else
            {
               k--;
               m++;
            }
         }
      }
   }
   if (m > 0)
      fprintf(stderr, "### Seeding error ###\n");
}

int RunModel(int run) // added run number as parameter
{
   int j, k, l, fs, fs2, nu, ni, nsi[MAX_NUM_SEED_LOCATIONS] /*Denotes either Num imported Infections given rate ir, or
                                                                number false positive "infections"*/
      ;
   double ir; // infection import rate?;
   double t, cI, lcI, t2;
   unsigned short int ts;
   int continueEvents = 1;

   /*	fprintf(stderr, "Checking consistency of initial state...\n");
      int i, i2, k2;
      for (i = j = k = ni = fs2 = i2 = 0; i < g_allParams.N; i++)
      {
         if (i % 1000 == 0) fprintf(stderr, "\r*** %i              ", i);
         if (Hosts[i].inf == 0) j++;
         if ((Hosts[i].pcell < g_allParams.NC) && (Hosts[i].pcell >= 0))
         {
            if (Cells[Hosts[i].pcell].susceptible[Hosts[i].listpos] != i)
            {
               k++;
               for (l = fs = 0; (l < Cells[Hosts[i].pcell].n) && (!fs); l++)
                  fs = (Cells[Hosts[i].pcell].susceptible[l] == i);
               if (!fs) ni++;
            }
            else
            {
               if ((Hosts[i].listpos > Cells[Hosts[i].pcell].S - 1) && (Hosts[i].inf == InfStat_Susceptible)) i2++;
               if ((Hosts[i].listpos < Cells[Hosts[i].pcell].S + Cells[Hosts[i].pcell].L + Cells[Hosts[i].pcell].I - 1)
      && (abs(Hosts[i].inf) == InfStat_Recovered)) i2++;
            }
            if ((Cells[Hosts[i].pcell].S + Cells[Hosts[i].pcell].L + Cells[Hosts[i].pcell].I + Cells[Hosts[i].pcell].R +
      Cells[Hosts[i].pcell].D) != Cells[Hosts[i].pcell].n)
            {
               k2++;
            }
         }
         else
            fs2++;
      }
      fprintf(stderr, "\n*** susceptibles=%i\nincorrect listpos=%i\nhosts not found in cell list=%i\nincorrect cell
      refs=%i\nincorrect positioning in cell susc list=%i\nwrong cell totals=%i\n", j, k, ni, fs2, i2, k2);
   */
   InterruptRun = 0;
   lcI          = 1;
   if (g_allParams.DoLoadSnapshot)
   {
      g_allParams.ts_age = (int)(g_allParams.SnapshotLoadTime * g_allParams.TimeStepsPerDay);
      t                  = ((double)g_allParams.ts_age) * g_allParams.TimeStep;
   }
   else
   {
      t                  = 0;
      g_allParams.ts_age = 0;
   }
   fs  = 1;
   fs2 = 0;
   nu  = 0;

   for (ns = 1; ((ns < g_allParams.totalSampleNumber) && (!InterruptRun)); ns++) //&&(continueEvents) <-removed this
   {
      RecordSample(t, ns - 1);
      fprintf(stderr, "\r    t=%lg   %i    %i|%i    %i     %i [=%i]  %i (%lg %lg %lg)   %lg    ", t, State.S, State.L,
              State.I, State.R, State.D, State.S + State.L + State.I + State.R + State.D, State.cumD, State.cumT,
              State.cumV, State.cumVG, sqrt(State.maxRad2) / 1000); // added State.cumVG
      if (!InterruptRun)
      {
         // Only run to a certain number of infections: ggilani 28/10/14
         if (g_allParams.LimitNumInfections)
            continueEvents = (State.cumI < g_allParams.MaxNumInfections);

         for (j = 0; ((j < g_allParams.UpdatesPerSample) && (!InterruptRun) && (continueEvents)); j++)
         {
            ts = (unsigned short int)(g_allParams.TimeStepsPerDay * t);

            // if we are to reset random numbers after an intervention event, specific time
            if (g_allParams.ResetSeedsPostIntervention)
               if ((g_allParams.ResetSeedsFlag == 0)
                   && (ts >= (g_allParams.TimeToResetSeeds * g_allParams.TimeStepsPerDay)))
               {
                  setall(&g_allParams.nextRunSeed1, &g_allParams.nextRunSeed2);
                  g_allParams.ResetSeedsFlag = 1;
               }

            if (fs)
            {
               if (g_allParams.DoAirports)
                  TravelDepartSweep(t);
               k = (int)t;
               if (g_allParams.DurImportTimeProfile > 0)
               {
                  if (k < g_allParams.DurImportTimeProfile)
                     ir = g_allParams.ImportInfectionTimeProfile[k]
                          * ((t > g_allParams.InfectionImportChangeTime)
                                ? (g_allParams.InfectionImportRate2 / g_allParams.InfectionImportRate1)
                                : 1.0);
                  else
                     ir = 0;
               }
               else
                  ir = (t > g_allParams.InfectionImportChangeTime) ? g_allParams.InfectionImportRate2
                                                                   : g_allParams.InfectionImportRate1;
               if (ir > 0) //// if infection import rate > 0, seed some infections
               {
                  for (k = ni = 0; k < g_allParams.NumSeedLocations; k++)
                     ni += (nsi[k] = (int)ignpoi(
                               g_allParams.TimeStep * ir * g_allParams.InitialInfectionsAdminUnitWeight[k]
                               * g_allParams
                                    .SeedingScaling)); //// sample number imported infections from Poisson distribution.
                  if (ni > 0)
                     SeedInfection(t, nsi, 1, run);
               }
               if (g_allParams.FalsePositivePerCapitaIncidence > 0)
               {
                  ni = (int)ignpoi(g_allParams.TimeStep * g_allParams.FalsePositivePerCapitaIncidence
                                   * ((double)g_allParams.populationSize));
                  if (ni > 0)
                  {
                     for (k = 0; k < ni; k++)
                     {
                        do
                        {
                           l = (int)(((double)g_allParams.populationSize)
                                     * ranf()); //// choose person l randomly from entire population. (but change l if
                                                ///while condition not satisfied?)
                        } while ((abs(Hosts[l].inf) == InfStat_Dead)
                                 || (ranf() > g_allParams.FalsePositiveAgeRate[HOST_AGE_GROUP(l)]));
                        DoFalseCase(l, t, ts, 0);
                     }
                  }
               }
               InfectSweep(t, run); // loops over all infectious people and decides which susceptible people to infect
                                    // (at household, place and spatial level), and adds them to queue. Then changes
                                    // each person's various characteristics using DoInfect function.  adding run number
                                    // as a parameter to infect sweep so we can track run number: ggilani - 15/10/14
               //// IncubRecoverySweep loops over all infecteds (either latent or infectious). If t is the right time,
               ///latent people moved to being infected, and infectious people moved to being clinical cases. Possibly
               ///also add them to recoveries or deaths. Add them to hospitalisation & hospitalisation discharge queues.
               if (!g_allParams.DoSI)
                  IncubRecoverySweep(t, run);
               // If doing new contact tracing, update numbers of people under contact tracing after each time step

               if (g_allParams.DoDigitalContactTracing)
                  DigitalContactTracingSweep(t);

               nu++;
               fs2 = ((g_allParams.DoDeath) || (State.L + State.I > 0) || (ir > 0)
                      || (g_allParams.FalsePositivePerCapitaIncidence > 0));
               ///// TreatSweep loops over microcells to decide which cells are treated (either with treatment, vaccine,
               ///social distancing, movement restrictions etc.). Calls DoVacc, DoPlaceClose, DoProphNoDelay etc. to
               ///change (threaded) State variables
               if (!TreatSweep(t))
               {
                  if ((!fs2) && (State.L + State.I == 0) && (g_allParams.FalsePositivePerCapitaIncidence == 0))
                  {
                     if ((ir == 0) && (((int)t) > g_allParams.DurImportTimeProfile))
                        fs = 0;
                  }
               }
               if (g_allParams.DoAirports)
                  TravelReturnSweep(t);
            }
            t += g_allParams.TimeStep;
            if (g_allParams.DoDeath)
               g_allParams.ts_age++;
            if ((g_allParams.DoSaveSnapshot) && (t <= g_allParams.SnapshotSaveTime)
                && (t + g_allParams.TimeStep > g_allParams.SnapshotSaveTime))
               SaveSnapshot();
            if (t > g_allParams.TreatNewCoursesStartTime)
               g_allParams.TreatMaxCourses += g_allParams.TimeStep * g_allParams.TreatNewCoursesRate;
            if ((t > g_allParams.VaccNewCoursesStartTime) && (t < g_allParams.VaccNewCoursesEndTime))
               g_allParams.VaccMaxCourses += g_allParams.TimeStep * g_allParams.VaccNewCoursesRate;
            cI = ((double)(State.S)) / ((double)g_allParams.populationSize);
            if ((lcI - cI) > 0.2)
            {
               lcI = cI;
               UpdateProbs(0);
               DoInitUpdateProbs = 1;
            }
         }
      }
   }
   if (!InterruptRun)
      RecordSample(t, g_allParams.totalSampleNumber - 1);
   fprintf(stderr, "\nEnd of run\n");
   t2 = t + g_allParams.SampleTime;
   //	if(!InterruptRun)
   while (fs)
   {
      fs = TreatSweep(t2);
      t2 += g_allParams.SampleStep;
   }
   //	fprintf(stderr,"End RunModel\n");
   if (g_allParams.DoAirports)
   {
      t2 = t;
      for (t2 = t; t2 <= t + MAX_TRAVEL_TIME; t2 += g_allParams.TimeStep)
         TravelReturnSweep(t2);
   }
   /*	if (!InterruptRun)
      {
         fprintf(stderr, "Checking consistency of final state...\n");
         int i, i2, k2;
         for (i = j = k = ni = fs2 = i2 = 0; i < g_allParams.N; i++)
         {
            if (i % 1000 == 0) fprintf(stderr, "\r*** %i              ", i);
            if (Hosts[i].inf == 0) j++;
            if ((Hosts[i].pcell < g_allParams.NC) && (Hosts[i].pcell >= 0))
            {
               if (Cells[Hosts[i].pcell].susceptible[Hosts[i].listpos] != i)
               {
                  k++;
                  for (l = fs = 0; (l < Cells[Hosts[i].pcell].n) && (!fs); l++)
                     fs = (Cells[Hosts[i].pcell].susceptible[l] == i);
                  if (!fs) ni++;
               }
               else
               {
                  if ((Hosts[i].listpos > Cells[Hosts[i].pcell].S - 1) && (Hosts[i].inf == InfStat_Susceptible)) i2++;
                  if ((Hosts[i].listpos < Cells[Hosts[i].pcell].S + Cells[Hosts[i].pcell].L + Cells[Hosts[i].pcell].I -
      1) && (abs(Hosts[i].inf) == InfStat_Recovered)) i2++;
               }
               if ((Cells[Hosts[i].pcell].S + Cells[Hosts[i].pcell].L + Cells[Hosts[i].pcell].I +
      Cells[Hosts[i].pcell].R + Cells[Hosts[i].pcell].D) != Cells[Hosts[i].pcell].n)
               {
                  k2++;
               }
            }
            else
               fs2++;
         }
         fprintf(stderr, "\n*** susceptibles=%i\nincorrect listpos=%i\nhosts not found in cell list=%i\nincorrect cell
      refs=%i\nincorrect positioning in cell susc list=%i\nwrong cell totals=%i\n", j, k, ni, fs2, i2, k2);
      }
   */
   if (!InterruptRun)
      RecordInfTypes();
   return (InterruptRun);
}

void SaveDistribs(void)
{
   int i, j, k;
   FILE* dat;
   char outname[1024];
   double s;

   if (g_allParams.DoPlaces)
   {
      for (j = 0; j < g_allParams.PlaceTypeNum; j++)
         if (j != g_allParams.HotelPlaceType)
         {
            for (i = 0; i < g_allParams.Nplace[j]; i++)
               Places[j][i].n = 0;
            for (i = 0; i < g_allParams.populationSize; i++)
            {
               if (Hosts[i].PlaceLinks[j] >= g_allParams.Nplace[j])
                  fprintf(stderr, "*%i %i: %i %i", i, j, Hosts[i].PlaceLinks[j], g_allParams.Nplace[j]);
               else if (Hosts[i].PlaceLinks[j] >= 0)
                  Places[j][Hosts[i].PlaceLinks[j]].n++;
            }
         }
      for (j = 0; j < g_allParams.PlaceTypeNum; j++)
         for (i = 0; i < MAX_DIST; i++)
            PlaceDistDistrib[j][i] = 0;
      for (i = 0; i < g_allParams.populationSize; i++)
         for (j = 0; j < g_allParams.PlaceTypeNum; j++)
            if ((j != g_allParams.HotelPlaceType) && (Hosts[i].PlaceLinks[j] >= 0))
            {
               if (Hosts[i].PlaceLinks[j] >= g_allParams.Nplace[j])
                  fprintf(stderr, "*%i %i: %i ", i, j, Hosts[i].PlaceLinks[j]);
               else if ((!g_allParams.DoOutputPlaceDistForOneAdunit)
                        || ((AdUnits[Mcells[Hosts[i].mcell].adunit].id % g_allParams.AdunitLevel1Mask)
                               / g_allParams.AdunitLevel1Divisor
                            == (g_allParams.OutputPlaceDistAdunit % g_allParams.AdunitLevel1Mask)
                                  / g_allParams.AdunitLevel1Divisor))
               {
                  k = Hosts[i].PlaceLinks[j];
                  s = sqrt(dist2_raw(Households[Hosts[i].hh].loc_x, Households[Hosts[i].hh].loc_y, Places[j][k].loc_x,
                                     Places[j][k].loc_y))
                      / OUTPUT_DIST_SCALE;
                  k = (int)s;
                  if (k < MAX_DIST)
                     PlaceDistDistrib[j][k]++;
               }
            }
      for (j = 0; j < g_allParams.PlaceTypeNum; j++)
         for (i = 0; i < MAX_PLACE_SIZE; i++)
            PlaceSizeDistrib[j][i] = 0;
      for (j = 0; j < g_allParams.PlaceTypeNum; j++)
         if (j != g_allParams.HotelPlaceType)
            for (i = 0; i < g_allParams.Nplace[j]; i++)
               if (Places[j][i].n < MAX_PLACE_SIZE)
                  PlaceSizeDistrib[j][Places[j][i].n]++;
      sprintf(outname, "%s.placedist.xls", OutFilePath);
      if (!(dat = fopen(outname, "wb")))
         ERR_CRITICAL("Unable to open output file\n");
      fprintf(dat, "dist");
      for (j = 0; j < g_allParams.PlaceTypeNum; j++)
         if (j != g_allParams.HotelPlaceType)
            fprintf(dat, "\tfreq_p%i", j);
      fprintf(dat, "\n");
      for (i = 0; i < MAX_DIST; i++)
      {
         fprintf(dat, "%i", i);
         for (j = 0; j < g_allParams.PlaceTypeNum; j++)
            if (j != g_allParams.HotelPlaceType)
               fprintf(dat, "\t%i", PlaceDistDistrib[j][i]);
         fprintf(dat, "\n");
      }
      fclose(dat);
      sprintf(outname, "%s.placesize.xls", OutFilePath);
      if (!(dat = fopen(outname, "wb")))
         ERR_CRITICAL("Unable to open output file\n");
      fprintf(dat, "size");
      for (j = 0; j < g_allParams.PlaceTypeNum; j++)
         if (j != g_allParams.HotelPlaceType)
            fprintf(dat, "\tfreq_p%i", j);
      fprintf(dat, "\n");
      for (i = 0; i < MAX_PLACE_SIZE; i++)
      {
         fprintf(dat, "%i", i);
         for (j = 0; j < g_allParams.PlaceTypeNum; j++)
            if (j != g_allParams.HotelPlaceType)
               fprintf(dat, "\t%i", PlaceSizeDistrib[j][i]);
         fprintf(dat, "\n");
      }
      fclose(dat);
   }
}

void SaveOriginDestMatrix(void)
{
   /** function: SaveOriginDestMatrix
    *
    * purpose: to save the calculated origin destination matrix to file
    * parameters: none
    * returns: none
    *
    * author: ggilani, 13/02/15
    */
   int i, j;
   FILE* dat;
   char outname[1024];

   sprintf(outname, "%s.origdestmat.xls", OutFilePath);
   if (!(dat = fopen(outname, "wb")))
      ERR_CRITICAL("Unable to open output file\n");
   fprintf(dat, "0,");
   for (i = 0; i < g_allParams.NumAdunits; i++)
      fprintf(dat, "%i,", (AdUnits[i].id % g_allParams.AdunitLevel1Mask) / g_allParams.AdunitLevel1Divisor);
   fprintf(dat, "\n");
   for (i = 0; i < g_allParams.NumAdunits; i++)
   {
      fprintf(dat, "%i,", (AdUnits[i].id % g_allParams.AdunitLevel1Mask) / g_allParams.AdunitLevel1Divisor);
      for (j = 0; j < g_allParams.NumAdunits; j++)
      {
         fprintf(dat, "%.10f,", AdUnits[i].origin_dest[j]);
      }
      fprintf(dat, "\n");
   }
   fclose(dat);
}

void SaveResults(void)
{
   int i, j;
   FILE* dat;
   char outname[1024];

   if (g_allParams.OutputNonSeverity)
   {
      sprintf(outname, "%s.xls", OutFilePath);
      if (!(dat = fopen(outname, "wb")))
         ERR_CRITICAL("Unable to open output file\n");
      fprintf(
         dat,
         "t\tS\tL\tI\tR\tD\tincI\tincR\tincFC\tincC\tincDC\tincTC\tincH\tincCT\tincCC\tcumT\tcumTP\tcumV\tcumVG\tExtinc"
         "t\trmsRad\tmaxRad\n"); //\t\t%.10f\t%.10f\t%.10f\n",g_allParams.R0household,g_allParams.R0places,g_allParams.R0spatial);
      for (i = 0; i < g_allParams.totalSampleNumber; i++)
      {
         fprintf(dat,
                 "%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%."
                 "10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\n",
                 TimeSeries[i].t, TimeSeries[i].S, TimeSeries[i].L, TimeSeries[i].I, TimeSeries[i].R, TimeSeries[i].D,
                 TimeSeries[i].incI, TimeSeries[i].incR, TimeSeries[i].incFC, TimeSeries[i].incC, TimeSeries[i].incDC,
                 TimeSeries[i].incTC, TimeSeries[i].incH, TimeSeries[i].incCT, TimeSeries[i].incCC, TimeSeries[i].cumT,
                 TimeSeries[i].cumTP, TimeSeries[i].cumV, TimeSeries[i].cumVG, TimeSeries[i].extinct,
                 TimeSeries[i].rmsRad, TimeSeries[i].maxRad);
      }
      fclose(dat);
   }

   if ((g_allParams.DoAdUnits) && (g_allParams.DoAdunitOutput))
   {
      sprintf(outname, "%s.adunit.xls", OutFilePath);
      if (!(dat = fopen(outname, "wb")))
         ERR_CRITICAL("Unable to open output file\n");
      fprintf(dat, "t");
      for (i = 0; i < g_allParams.NumAdunits; i++)
         fprintf(dat, "\tI_%s", AdUnits[i].ad_name);
      for (i = 0; i < g_allParams.NumAdunits; i++)
         fprintf(dat, "\tC_%s", AdUnits[i].ad_name);
      for (i = 0; i < g_allParams.NumAdunits; i++)
         fprintf(dat, "\tDC_%s", AdUnits[i].ad_name);

      fprintf(dat, "\n");
      for (i = 0; i < g_allParams.totalSampleNumber; i++)
      {
         fprintf(dat, "%.10f", TimeSeries[i].t);
         for (j = 0; j < g_allParams.NumAdunits; j++)
            fprintf(dat, "\t%.10f", TimeSeries[i].incI_adunit[j]);
         for (j = 0; j < g_allParams.NumAdunits; j++)
            fprintf(dat, "\t%.10f", TimeSeries[i].incC_adunit[j]);
         for (j = 0; j < g_allParams.NumAdunits; j++)
            fprintf(dat, "\t%.10f", TimeSeries[i].incDC_adunit[j]);
         fprintf(dat, "\n");
      }
      fclose(dat);
   }

   if ((g_allParams.DoDigitalContactTracing) && (g_allParams.DoAdUnits) && (g_allParams.OutputDigitalContactTracing))
   {
      sprintf(outname, "%s.digitalcontacttracing.xls", OutFilePath); // modifying to csv file
      if (!(dat = fopen(outname, "wb")))
         ERR_CRITICAL("Unable to open output file\n");
      fprintf(dat, "t");
      for (i = 0; i < g_allParams.NumAdunits; i++)
      {
         fprintf(dat, "\tincDCT_%s", AdUnits[i].ad_name);
      }
      for (i = 0; i < g_allParams.NumAdunits; i++)
      {
         fprintf(dat, "\tDCT_%s", AdUnits[i].ad_name);
      }
      fprintf(dat, "\n");
      // print actual output
      for (i = 0; i < g_allParams.totalSampleNumber; i++)
      {
         fprintf(dat, "%.10lf", TimeSeries[i].t);
         for (j = 0; j < g_allParams.NumAdunits; j++)
         {
            fprintf(dat, "\t%.10lf", TimeSeries[i].incDCT_adunit[j]);
         }
         for (j = 0; j < g_allParams.NumAdunits; j++)
         {
            fprintf(dat, "\t%.10lf", TimeSeries[i].DCT_adunit[j]);
         }
         fprintf(dat, "\n");
      }
      fclose(dat);
   }

   if ((g_allParams.DoDigitalContactTracing) && (g_allParams.OutputDigitalContactDist))
   {
      sprintf(outname, "%s.digitalcontactdist.xls", OutFilePath); // modifying to csv file
      if (!(dat = fopen(outname, "wb")))
         ERR_CRITICAL("Unable to open output file\n");
      // print headers
      fprintf(dat, "nContacts\tFrequency\n");
      for (i = 0; i < (MAX_CONTACTS + 1); i++)
      {
         fprintf(dat, "%i\t%i\n", i, State.contact_dist[i]);
      }
      fclose(dat);
   }

   if (g_allParams.KeyWorkerProphTimeStartBase < g_allParams.SampleTime)
   {
      sprintf(outname, "%s.keyworker.xls", OutFilePath);
      if (!(dat = fopen(outname, "wb")))
         ERR_CRITICAL("Unable to open output file\n");
      fprintf(dat, "t");
      for (i = 0; i < 2; i++)
         fprintf(dat, "\tI%i", i);
      for (i = 0; i < 2; i++)
         fprintf(dat, "\tC%i", i);
      for (i = 0; i < 2; i++)
         fprintf(dat, "\tT%i", i);
      fprintf(dat, "\t%i\t%i\n", g_allParams.KeyWorkerNum, g_allParams.KeyWorkerIncHouseNum);
      for (i = 0; i < g_allParams.totalSampleNumber; i++)
      {
         fprintf(dat, "%.10f", TimeSeries[i].t);
         for (j = 0; j < 2; j++)
            fprintf(dat, "\t%.10f", TimeSeries[i].incI_keyworker[j]);
         for (j = 0; j < 2; j++)
            fprintf(dat, "\t%.10f", TimeSeries[i].incC_keyworker[j]);
         for (j = 0; j < 2; j++)
            fprintf(dat, "\t%.10f", TimeSeries[i].cumT_keyworker[j]);
         fprintf(dat, "\n");
      }
      fclose(dat);
   }

   if (g_allParams.DoInfectionTree)
   {
      sprintf(outname, "%s.tree.xls", OutFilePath);
      if (!(dat = fopen(outname, "wb")))
         ERR_CRITICAL("Unable to open output file\n");
      for (i = 0; i < g_allParams.populationSize; i++)
         if (Hosts[i].infect_type % INFECT_TYPE_MASK > 0)
            fprintf(dat, "%i\t%i\t%i\t%i\n", i, Hosts[i].infector, Hosts[i].infect_type % INFECT_TYPE_MASK,
                    (int)HOST_AGE_YEAR(i));
      fclose(dat);
   }
#if defined(WIN32_BM) || defined(IMAGE_MAGICK)
   static int dm[13] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
   int d, m, y, dml, f;
#ifdef WIN32_BM
   // if(g_allParams.OutputBitmap == 1) CloseAvi(avi);
   // if((TimeSeries[g_allParams.NumSamples - 1].extinct) && (g_allParams.OutputOnlyNonExtinct))
   //	{
   //	sprintf(outname, "%s.ge" DIRECTORY_SEPARATOR "%s.avi", OutFile, OutFile);
   //	DeleteFile(outname);
   //	}
#endif
   if (g_allParams.OutputBitmap >= 1)
   {
      // Generate Google Earth .kml file
      sprintf(outname, "%s.ge" DIRECTORY_SEPARATOR "%s.ge.kml", OutFilePath,
              OutFilePath); // sprintf(outname,"%s.ge" DIRECTORY_SEPARATOR "%s.kml",OutFileBase,OutFile);
      if (!(dat = fopen(outname, "wb")))
      {
         ERR_CRITICAL("Unable to open output kml file\n");
      }
      fprintf(
         dat,
         "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<kml xmlns=\"http://earth.google.com/kml/2.2\">\n<Document>\n");
      fprintf(dat, "<name>%s</name>\n", OutFilePath);
      y = 2009;
      m = 1;
      d = 1;
      for (i = 0; i < g_allParams.totalSampleNumber; i++)
      {
         fprintf(dat, "<GroundOverlay>\n<name>Snapshot %i</name>\n", i + 1);
         fprintf(dat, "<TimeSpan>\n<begin>%i-%02i-%02iT00:00:00Z</begin>\n", y, m, d);
         d += (int)g_allParams.SampleStep; // SampleStep has to be an integer here.
         do
         {
            f   = 1;
            dml = dm[m];
            if ((m == 2) && (y % 4 == 0))
               dml = 29;
            if (d > dml)
            {
               m++;
               if (m > 12)
               {
                  m -= 12;
                  y++;
               }
               d -= dml;
               f = 0;
            }
         } while (!f);
         fprintf(dat, "<end>%i-%02i-%02iT00:00:00Z</end>\n</TimeSpan>\n", y, m, d);
         sprintf(outname, "%s.ge" DIRECTORY_SEPARATOR "%s.%i.png", OutFilePath, OutFilePath, i + 1);
         fprintf(dat, "<Icon>\n<href>%s</href>\n</Icon>\n", outname);
         fprintf(dat,
                 "<LatLonBox>\n<north>%.10f</north>\n<south>%.10f</south>\n<east>%.10f</east>\n<west>%.10f</west>\n</"
                 "LatLonBox>\n",
                 g_allParams.SpatialBoundingBox[3], g_allParams.SpatialBoundingBox[1],
                 g_allParams.SpatialBoundingBox[2], g_allParams.SpatialBoundingBox[0]);
         fprintf(dat, "</GroundOverlay>\n");
      }
      fprintf(dat, "</Document>\n</kml>\n");
      fclose(dat);
   }
#endif

   if (g_allParams.DoSeverity)
   {
      sprintf(outname, "%s.severity.xls", OutFilePath);
      if (!(dat = fopen(outname, "wb")))
         ERR_CRITICAL("Unable to open severity output file\n");
      fprintf(
         dat,
         "t\tS\tI\tR\tincI\tMild\tILI\tSARI\tCritical\tCritRecov\tincMild\tincILI\tincSARI\tincCritical\tincCritRecov\t"
         "incDeath\tincDeath_ILI\tincDeath_SARI\tincDeath_"
         "Critical\tcumMild\tcumILI\tcumSARI\tcumCritical\tcumCritRecov\tcumDeath\tcumDeath_ILI\tcumDeath_"
         "SARI\tcumDeath_Critical\n"); //\t\t%.10f\t%.10f\t%.10f\n",g_allParams.R0household,g_allParams.R0places,g_allParams.R0spatial);
      for (i = 0; i < g_allParams.totalSampleNumber; i++)
      {
         fprintf(dat,
                 "%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%."
                 "10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\n",
                 TimeSeries[i].t, TimeSeries[i].S, TimeSeries[i].I, TimeSeries[i].R, TimeSeries[i].incI,
                 TimeSeries[i].Mild, TimeSeries[i].ILI, TimeSeries[i].SARI, TimeSeries[i].Critical,
                 TimeSeries[i].CritRecov, TimeSeries[i].incMild, TimeSeries[i].incILI, TimeSeries[i].incSARI,
                 TimeSeries[i].incCritical, TimeSeries[i].incCritRecov, TimeSeries[i].incD, TimeSeries[i].incDeath_ILI,
                 TimeSeries[i].incDeath_SARI, TimeSeries[i].incDeath_Critical, TimeSeries[i].cumMild,
                 TimeSeries[i].cumILI, TimeSeries[i].cumSARI, TimeSeries[i].cumCritical, TimeSeries[i].cumCritRecov,
                 TimeSeries[i].D, TimeSeries[i].cumDeath_ILI, TimeSeries[i].cumDeath_SARI,
                 TimeSeries[i].cumDeath_Critical);
      }
      fclose(dat);

      if ((g_allParams.DoAdUnits) && (g_allParams.OutputSeverityAdminUnit))
      {
         //// output severity results by admin unit
         sprintf(outname, "%s.severity.adunit.xls", OutFilePath);
         if (!(dat = fopen(outname, "wb")))
            ERR_CRITICAL("Unable to open output file\n");
         fprintf(dat, "t");

         /////// ****** /////// ****** /////// ****** COLNAMES
         //// prevalence
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tMild_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tILI_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tSARI_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tCritical_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tCritRecov_%s", AdUnits[i].ad_name);

         //// incidence
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tincI_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tincMild_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tincILI_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tincSARI_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tincCritical_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tincCritRecov_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tincDeath_adu%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tincDeath_ILI_adu%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tincDeath_SARI_adu%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tincDeath_Critical_adu%s", AdUnits[i].ad_name);

         //// cumulative incidence
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tcumMild_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tcumILI_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tcumSARI_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tcumCritical_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tcumCritRecov_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tcumDeaths_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tcumDeath_ILI_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tcumDeath_SARI_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tcumDeath_Critical_%s", AdUnits[i].ad_name);

         fprintf(dat, "\n");

         /////// ****** /////// ****** /////// ****** Populate table.
         for (i = 0; i < g_allParams.totalSampleNumber; i++)
         {
            fprintf(dat, "%.10f", TimeSeries[i].t);

            //// prevalence
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", TimeSeries[i].Mild_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", TimeSeries[i].ILI_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", TimeSeries[i].SARI_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", TimeSeries[i].Critical_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", TimeSeries[i].CritRecov_adunit[j]);

            //// incidence
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", TimeSeries[i].incI_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", TimeSeries[i].incMild_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", TimeSeries[i].incILI_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", TimeSeries[i].incSARI_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", TimeSeries[i].incCritical_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", TimeSeries[i].incCritRecov_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", TimeSeries[i].incD_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", TimeSeries[i].incDeath_ILI_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", TimeSeries[i].incDeath_SARI_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", TimeSeries[i].incDeath_Critical_adunit[j]);

            //// cumulative incidence
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", TimeSeries[i].cumMild_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", TimeSeries[i].cumILI_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", TimeSeries[i].cumSARI_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", TimeSeries[i].cumCritical_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", TimeSeries[i].cumCritRecov_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", TimeSeries[i].cumD_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", TimeSeries[i].cumDeath_ILI_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", TimeSeries[i].cumDeath_SARI_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", TimeSeries[i].cumDeath_Critical_adunit[j]);

            if (i != g_allParams.totalSampleNumber - 1)
               fprintf(dat, "\n");
         }
         fclose(dat);
      }
   }
}

void SaveSummaryResults(void) //// calculates and saves summary results (called for average of extinct and non-extinct
                              ///realisation time series - look in main)
{
   int i, j;
   double c, t;
   FILE* dat;
   char outname[1024];

   c = 1 / ((double)(g_allParams.NRactE + g_allParams.NRactNE));

   if (g_allParams.OutputNonSeverity)
   {
      sprintf(outname, "%s.xls", OutFilePath);
      if (!(dat = fopen(outname, "wb")))
         ERR_CRITICAL("Unable to open output file\n");
      //// set colnames
      fprintf(dat,
              "t\tS\tL\tI\tR\tD\tincI\tincR\tincD\tincC\tincDC\tincTC\tincH\tcumT\tcumTmax\tcumTP\tcumV\tcumVmax\tExtin"
              "ct\trmsRad\tmaxRad\tvS\tvI\tvR\tvD\tvincI\tvincR\tvincFC\tvincC\tvincDC\tvincTC\tvincH\tvrmsRad\tvmaxRad"
              "\t\t%i\t%i\t%.10f\t%.10f\t%.10f\t\t%.10f\t%.10f\t%.10f\t%.10f\n",
              g_allParams.NRactNE, g_allParams.NRactE, g_allParams.R0household, g_allParams.R0places,
              g_allParams.R0spatial, c * PeakHeightSum, c * PeakHeightSS - c * c * PeakHeightSum * PeakHeightSum,
              c * PeakTimeSum, c * PeakTimeSS - c * c * PeakTimeSum * PeakTimeSum);
      c = 1 / ((double)g_allParams.NRactual);

      //// populate table
      for (i = 0; i < g_allParams.totalSampleNumber; i++)
      {
         fprintf(dat,
                 "%.10f\t%10lf\t%10lf\t%10lf\t%10lf\t%10lf\t%10lf\t%10lf\t%10lf\t%10lf\t%10lf\t%10lf\t%10lf\t%10lf\t%"
                 "10lf\t%10lf\t%10lf\t%10lf\t%10lf\t%10lf\t%10lf\t",
                 c * TSMean[i].t, c * TSMean[i].S, c * TSMean[i].L, c * TSMean[i].I, c * TSMean[i].R, c * TSMean[i].D,
                 c * TSMean[i].incI, c * TSMean[i].incR, c * TSMean[i].incFC, c * TSMean[i].incC, c * TSMean[i].incDC,
                 c * TSMean[i].incTC, c * TSMean[i].incH, c * TSMean[i].cumT, TSMean[i].cumTmax, c * TSMean[i].cumTP,
                 c * TSMean[i].cumV, TSMean[i].cumVmax, c * TSMean[i].extinct, c * TSMean[i].rmsRad,
                 c * TSMean[i].maxRad);
         fprintf(dat, "%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\n",
                 c * TSVar[i].S - c * c * TSMean[i].S * TSMean[i].S, c * TSVar[i].I - c * c * TSMean[i].I * TSMean[i].I,
                 c * TSVar[i].R - c * c * TSMean[i].R * TSMean[i].R, c * TSVar[i].D - c * c * TSMean[i].D * TSMean[i].D,
                 c * TSVar[i].incI - c * c * TSMean[i].incI * TSMean[i].incI,
                 c * TSVar[i].incR - c * c * TSMean[i].incR * TSMean[i].incR,
                 c * TSVar[i].incD - c * c * TSMean[i].incD * TSMean[i].incD,
                 c * TSVar[i].incC - c * c * TSMean[i].incC * TSMean[i].incC,
                 c * TSVar[i].incDC - c * c * TSMean[i].incDC * TSMean[i].incDC, // added detected cases
                 c * TSVar[i].incTC - c * c * TSMean[i].incTC * TSMean[i].incTC,
                 c * TSVar[i].incH - c * c * TSMean[i].incH * TSMean[i].incH, // added hospitalisation
                 c * TSVar[i].rmsRad - c * c * TSMean[i].rmsRad * TSMean[i].rmsRad,
                 c * TSVar[i].maxRad - c * c * TSMean[i].maxRad * TSMean[i].maxRad);
      }
      fclose(dat);
   }

   if (g_allParams.OutputControls)
   {
      sprintf(outname, "%s.controls.xls", OutFilePath);
      if (!(dat = fopen(outname, "wb")))
         ERR_CRITICAL("Unable to open output file\n");
      fprintf(dat, "t\tS\tincC\tincTC\tincFC\tincH\tcumT\tcumUT\tcumTP\tcumV\tincHQ\tincAC\tincAH\tincAA\tincACS\tincAP"
                   "C\tincAPA\tincAPCS\tpropSocDist");
      for (j = 0; j < NUM_PLACE_TYPES; j++)
         fprintf(dat, "\tprClosed_%i", j);
      fprintf(dat, "t\tvS\tvincC\tvincTC\tvincFC\tvincH\tvcumT\tvcumUT\tvcumTP\tvcumV");
      for (j = 0; j < NUM_PLACE_TYPES; j++)
         fprintf(dat, "\tvprClosed_%i", j);
      fprintf(dat, "\n");
      for (i = 0; i < g_allParams.totalSampleNumber; i++)
      {
         fprintf(dat, "%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf",
                 c * TSMean[i].t, c * TSMean[i].S, c * TSMean[i].incC, c * TSMean[i].incTC, c * TSMean[i].incFC,
                 c * TSMean[i].incH, c * TSMean[i].cumT, c * TSMean[i].cumUT, c * TSMean[i].cumTP, c * TSMean[i].cumV,
                 c * TSMean[i].incHQ, c * TSMean[i].incAC, c * TSMean[i].incAH, c * TSMean[i].incAA,
                 c * TSMean[i].incACS, c * TSMean[i].incAPC, c * TSMean[i].incAPA, c * TSMean[i].incAPCS,
                 c * TSMean[i].PropSocDist);
         for (j = 0; j < NUM_PLACE_TYPES; j++)
            fprintf(dat, "\t%lf", c * TSMean[i].PropPlacesClosed[j]);
         fprintf(dat, "\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf",
                 c * TSVar[i].S - c * c * TSMean[i].S * TSMean[i].S,
                 c * TSVar[i].incC - c * c * TSMean[i].incC * TSMean[i].incC,
                 c * TSVar[i].incTC - c * c * TSMean[i].incTC * TSMean[i].incTC,
                 c * TSVar[i].incFC - c * c * TSMean[i].incFC * TSMean[i].incFC,
                 c * TSVar[i].incH - c * c * TSMean[i].incH * TSMean[i].incH,
                 c * TSVar[i].cumT - c * c * TSMean[i].cumT * TSMean[i].cumT,
                 c * TSVar[i].cumUT - c * c * TSMean[i].cumUT * TSMean[i].cumUT,
                 c * TSVar[i].cumTP - c * c * TSMean[i].cumTP * TSMean[i].cumTP,
                 c * TSVar[i].cumV - c * c * TSMean[i].cumV * TSMean[i].cumV);
         for (j = 0; j < NUM_PLACE_TYPES; j++)
            fprintf(dat, "\t%lf", TSVar[i].PropPlacesClosed[j]);
         fprintf(dat, "\n");
      }
      fclose(dat);
   }

   if (g_allParams.OutputAge)
   {
      sprintf(outname, "%s.age.xls", OutFilePath);
      if (!(dat = fopen(outname, "wb")))
         ERR_CRITICAL("Unable to open output file\n");
      fprintf(dat, "t");
      for (i = 0; i < NUM_AGE_GROUPS; i++)
         fprintf(dat, "\tI%i-%i", AGE_GROUP_WIDTH * i, AGE_GROUP_WIDTH * (i + 1));
      for (i = 0; i < NUM_AGE_GROUPS; i++)
         fprintf(dat, "\tC%i-%i", AGE_GROUP_WIDTH * i, AGE_GROUP_WIDTH * (i + 1));
      for (i = 0; i < NUM_AGE_GROUPS; i++)
         fprintf(dat, "\tD%i-%i", AGE_GROUP_WIDTH * i, AGE_GROUP_WIDTH * (i + 1));
      fprintf(dat, "\n");
      for (i = 0; i < g_allParams.totalSampleNumber; i++)
      {
         fprintf(dat, "%.10f", c * TSMean[i].t);
         for (j = 0; j < NUM_AGE_GROUPS; j++)
            fprintf(dat, "\t%.10f", c * TSMean[i].incIa[j]);
         for (j = 0; j < NUM_AGE_GROUPS; j++)
            fprintf(dat, "\t%.10f", c * TSMean[i].incCa[j]);
         for (j = 0; j < NUM_AGE_GROUPS; j++)
            fprintf(dat, "\t%.10f", c * TSMean[i].incDa[j]);
         fprintf(dat, "\n");
      }
      fprintf(dat, "dist");
      for (j = 0; j < NUM_AGE_GROUPS; j++)
         fprintf(dat, "\t%.10f", AgeDist[j]);
      fprintf(dat, "\n");
      fclose(dat);
   }

   if ((g_allParams.DoAdUnits) && (g_allParams.DoAdunitOutput))
   {
      sprintf(outname, "%s.adunit.xls", OutFilePath);
      if (!(dat = fopen(outname, "wb")))
         ERR_CRITICAL("Unable to open output file\n");
      fprintf(dat, "t");
      for (i = 0; i < g_allParams.NumAdunits; i++)
         fprintf(dat, "\tI_%s", AdUnits[i].ad_name);
      for (i = 0; i < g_allParams.NumAdunits; i++)
         fprintf(dat, "\tC_%s", AdUnits[i].ad_name);
      for (i = 0; i < g_allParams.NumAdunits; i++)
         fprintf(dat, "\tDC_%s", AdUnits[i].ad_name); // added detected cases: ggilani 03/02/15
      for (i = 0; i < g_allParams.NumAdunits; i++)
         fprintf(dat, "\tT_%s", AdUnits[i].ad_name);
      for (i = 0; i < g_allParams.NumAdunits; i++)
         fprintf(dat, "\t%.10f", g_allParams.PopByAdunit[i][0]);
      for (i = 0; i < g_allParams.NumAdunits; i++)
         fprintf(dat, "\t%.10f", g_allParams.PopByAdunit[i][1]);
      fprintf(dat, "\n");
      for (i = 0; i < g_allParams.totalSampleNumber; i++)
      {
         fprintf(dat, "%.10f", c * TSMean[i].t);
         for (j = 0; j < g_allParams.NumAdunits; j++)
            fprintf(dat, "\t%.10f", c * TSMean[i].incI_adunit[j]);
         for (j = 0; j < g_allParams.NumAdunits; j++)
            fprintf(dat, "\t%.10f", c * TSMean[i].incC_adunit[j]);
         for (j = 0; j < g_allParams.NumAdunits; j++)
            fprintf(dat, "\t%.10f", c * TSMean[i].incDC_adunit[j]); // added detected cases: ggilani 03/02/15
         for (j = 0; j < g_allParams.NumAdunits; j++)
            fprintf(dat, "\t%.10f", c * TSMean[i].cumT_adunit[j]);
         fprintf(dat, "\n");
      }
      fclose(dat);

      if (g_allParams.OutputAdUnitVar)
      {
         sprintf(outname, "%s.adunitVar.xls", OutFilePath);
         if (!(dat = fopen(outname, "wb")))
            ERR_CRITICAL("Unable to open output file\n");
         fprintf(dat, "t");
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tI_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tC_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tDC_%s", AdUnits[i].ad_name); // added detected cases: ggilani 03/02/15
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tT_%s", AdUnits[i].ad_name);
         fprintf(dat, "\n");
         for (i = 0; i < g_allParams.totalSampleNumber; i++)
         {
            fprintf(dat, "%.10f", c * TSMean[i].t);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f",
                       c * TSVar[i].incI_adunit[j] - c * c * TSMean[i].incI_adunit[j] * TSMean[i].incI_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f",
                       c * TSVar[i].incC_adunit[j] - c * c * TSMean[i].incC_adunit[j] * TSMean[i].incC_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f",
                       c * TSVar[i].incDC_adunit[j]
                          - c * c * TSMean[i].incDC_adunit[j]
                               * TSMean[i].incDC_adunit[j]); // added detected cases: ggilani 03/02/15
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f",
                       c * TSVar[i].cumT_adunit[j] - c * c * TSMean[i].cumT_adunit[j] * TSMean[i].cumT_adunit[j]);
            fprintf(dat, "\n");
         }
         fclose(dat);
      }
   }

   if ((g_allParams.DoDigitalContactTracing) && (g_allParams.DoAdUnits) && (g_allParams.OutputDigitalContactTracing))
   {
      sprintf(outname, "%s.digitalcontacttracing.xls", OutFilePath);
      if (!(dat = fopen(outname, "wb")))
         ERR_CRITICAL("Unable to open output file\n");
      fprintf(dat, "t");
      for (i = 0; i < g_allParams.NumAdunits; i++)
      {
         fprintf(dat, "\tincDCT_%s", AdUnits[i].ad_name); // //printing headers for inc per admin unit
      }
      for (i = 0; i < g_allParams.NumAdunits; i++)
      {
         fprintf(dat, "\tDCT_%s",
                 AdUnits[i].ad_name); // //printing headers for prevalence of digital contact tracing per admin unit
      }
      fprintf(dat, "\n");
      // print actual output
      for (i = 0; i < g_allParams.totalSampleNumber; i++)
      {
         fprintf(dat, "%.10lf", c * TSMean[i].t);
         for (j = 0; j < g_allParams.NumAdunits; j++)
         {
            fprintf(dat, "\t%.10lf", c * TSMean[i].incDCT_adunit[j]);
         }
         for (j = 0; j < g_allParams.NumAdunits; j++)
         {
            fprintf(dat, "\t%.10lf", c * TSMean[i].DCT_adunit[j]);
         }
         fprintf(dat, "\n");
      }

      fclose(dat);
   }

   if (g_allParams.KeyWorkerProphTimeStartBase < g_allParams.SampleTime)
   {
      sprintf(outname, "%s.keyworker.xls", OutFilePath);
      if (!(dat = fopen(outname, "wb")))
         ERR_CRITICAL("Unable to open output file\n");
      fprintf(dat, "t");
      for (i = 0; i < 2; i++)
         fprintf(dat, "\tI%i", i);
      for (i = 0; i < 2; i++)
         fprintf(dat, "\tC%i", i);
      for (i = 0; i < 2; i++)
         fprintf(dat, "\tT%i", i);
      for (i = 0; i < 2; i++)
         fprintf(dat, "\tvI%i", i);
      for (i = 0; i < 2; i++)
         fprintf(dat, "\tvC%i", i);
      for (i = 0; i < 2; i++)
         fprintf(dat, "\tvT%i", i);
      fprintf(dat, "\t%i\t%i\n", g_allParams.KeyWorkerNum, g_allParams.KeyWorkerIncHouseNum);
      for (i = 0; i < g_allParams.totalSampleNumber; i++)
      {
         fprintf(dat, "%.10f", c * TSMean[i].t);
         for (j = 0; j < 2; j++)
            fprintf(dat, "\t%.10f", c * TSMean[i].incI_keyworker[j]);
         for (j = 0; j < 2; j++)
            fprintf(dat, "\t%.10f", c * TSMean[i].incC_keyworker[j]);
         for (j = 0; j < 2; j++)
            fprintf(dat, "\t%.10f", c * TSMean[i].cumT_keyworker[j]);
         for (j = 0; j < 2; j++)
            fprintf(dat, "\t%.10f",
                    c * TSVar[i].incI_keyworker[j] - c * c * TSMean[i].incI_keyworker[j] * TSMean[i].incI_keyworker[j]);
         for (j = 0; j < 2; j++)
            fprintf(dat, "\t%.10f",
                    c * TSVar[i].incC_keyworker[j] - c * c * TSMean[i].incC_keyworker[j] * TSMean[i].incC_keyworker[j]);
         for (j = 0; j < 2; j++)
            fprintf(dat, "\t%.10f",
                    c * TSVar[i].cumT_keyworker[j] - c * c * TSMean[i].cumT_keyworker[j] * TSMean[i].cumT_keyworker[j]);
         fprintf(dat, "\n");
      }
      fclose(dat);
   }

   if (g_allParams.OutputInfType)
   {
      sprintf(outname, "%s.inftype.xls", OutFilePath);
      if (!(dat = fopen(outname, "wb")))
         ERR_CRITICAL("Unable to open output file\n");
      fprintf(dat, "t\tR");
      for (j = 0; j < INFECT_TYPE_MASK; j++)
         fprintf(dat, "\tRtype_%i", j);
      for (j = 0; j < INFECT_TYPE_MASK; j++)
         fprintf(dat, "\tincItype_%i", j);
      for (j = 0; j < NUM_AGE_GROUPS; j++)
         fprintf(dat, "\tRage_%i", j);
      fprintf(dat, "\n");
      for (i = 0; i < g_allParams.totalSampleNumber; i++)
      {
         fprintf(dat, "%lf\t%lf", c * TSMean[i].t, c * TSMean[i].Rdenom);
         for (j = 0; j < INFECT_TYPE_MASK; j++)
            fprintf(dat, "\t%lf", c * TSMean[i].Rtype[j]);
         for (j = 0; j < INFECT_TYPE_MASK; j++)
            fprintf(dat, "\t%lf", c * TSMean[i].incItype[j]);
         for (j = 0; j < NUM_AGE_GROUPS; j++)
            fprintf(dat, "\t%lf", c * TSMean[i].Rage[j]);
         fprintf(dat, "\n");
      }
      fclose(dat);
   }

   if (g_allParams.OutputR0)
   {
      sprintf(outname, "%s.R0.xls", OutFilePath);
      if (!(dat = fopen(outname, "wb")))
         ERR_CRITICAL("Unable to open output file\n");
      for (i = 0; i < MAX_SEC_REC; i++)
      {
         fprintf(dat, "%i", i);
         for (j = 0; j < MAX_GEN_REC; j++)
            fprintf(dat, "\t%.10f", c * indivR0_av[i][j]);
         fprintf(dat, "\n");
      }
      fclose(dat);
   }

   if (g_allParams.OutputHousehold)
   {
      sprintf(outname, "%s.household.xls", OutFilePath);
      for (i = 1; i <= MAX_HOUSEHOLD_SIZE; i++)
      {
         t = 0;
         for (j = 1; j <= MAX_HOUSEHOLD_SIZE; j++)
            t += inf_household_av[i][j];
         inf_household_av[i][0] = denom_household[i] / c - t;
      }
      for (i = 1; i <= MAX_HOUSEHOLD_SIZE; i++)
      {
         t = 0;
         for (j = 1; j <= MAX_HOUSEHOLD_SIZE; j++)
            t += case_household_av[i][j];
         case_household_av[i][0] = denom_household[i] / c - t;
      }
      if (!(dat = fopen(outname, "wb")))
         ERR_CRITICAL("Unable to open output file\n");
      for (i = 1; i <= MAX_HOUSEHOLD_SIZE; i++)
         fprintf(dat, "\t%i", i);
      fprintf(dat, "\n");
      for (i = 0; i <= MAX_HOUSEHOLD_SIZE; i++)
      {
         fprintf(dat, "%i", i);
         for (j = 1; j <= MAX_HOUSEHOLD_SIZE; j++)
            fprintf(dat, "\t%.10f", inf_household_av[j][i] * c);
         fprintf(dat, "\n");
      }
      fprintf(dat, "\n");
      for (i = 1; i <= MAX_HOUSEHOLD_SIZE; i++)
         fprintf(dat, "\t%i", i);
      fprintf(dat, "\n");
      for (i = 0; i <= MAX_HOUSEHOLD_SIZE; i++)
      {
         fprintf(dat, "%i", i);
         for (j = 1; j <= MAX_HOUSEHOLD_SIZE; j++)
            fprintf(dat, "\t%.10f", case_household_av[j][i] * c);
         fprintf(dat, "\n");
      }
      fclose(dat);
   }

   if (g_allParams.OutputCountry)
   {
      sprintf(outname, "%s.country.xls", OutFilePath);
      if (!(dat = fopen(outname, "wb")))
         ERR_CRITICAL("Unable to open output file\n");
      for (i = 0; i < MAX_COUNTRIES; i++)
         fprintf(dat, "%i\t%.10f\t%.10f\n", i, infcountry_av[i] * c, infcountry_num[i] * c);
      fclose(dat);
   }

   if (g_allParams.DoSeverity)
   {
      //// output separate severity file (can integrate with main if need be)
      sprintf(outname, "%s.severity.xls", OutFilePath);

      if (!(dat = fopen(outname, "wb")))
         ERR_CRITICAL("Unable to open severity output file\n");
      fprintf(
         dat,
         "t\tPropSocDist\tS\tI\tR\tincI\tincC\tMild\tILI\tSARI\tCritical\tCritRecov\tSARIP\tCriticalP\tCritRecovP\tincM"
         "ild\tincILI\tincSARI\tincCritical\tincCritRecov\tincSARIP\tincCriticalP\tincCritRecovP\tincDeath\tincDeath_"
         "ILI\tincDeath_SARI\tincDeath_"
         "Critical\tcumMild\tcumILI\tcumSARI\tcumCritical\tcumCritRecov\tcumDeath\tcumDeath_ILI\tcumDeath_"
         "SARI\tcumDeath_Critical\n"); //\t\t%.10f\t%.10f\t%.10f\n",g_allParams.R0household,g_allParams.R0places,g_allParams.R0spatial);
      double SARI, Critical, CritRecov, incSARI, incCritical, incCritRecov, sc1, sc2, sc3,
         sc4; // this stuff corrects bed prevalence for exponentially distributed time to test results in hospital
      sc1 = (g_allParams.Mean_TimeToTest > 0) ? exp(-1.0 / g_allParams.Mean_TimeToTest) : 0.0;
      sc2 = (g_allParams.Mean_TimeToTest > 0) ? exp(-g_allParams.Mean_TimeToTestOffset / g_allParams.Mean_TimeToTest)
                                              : 0.0;
      sc3 = (g_allParams.Mean_TimeToTest > 0)
               ? exp(-g_allParams.Mean_TimeToTestCriticalOffset / g_allParams.Mean_TimeToTest)
               : 0.0;
      sc4 = (g_allParams.Mean_TimeToTest > 0)
               ? exp(-g_allParams.Mean_TimeToTestCritRecovOffset / g_allParams.Mean_TimeToTest)
               : 0.0;
      incSARI = incCritical = incCritRecov = 0;
      for (i = 0; i < g_allParams.totalSampleNumber; i++)
      {
         if (i > 0)
         {
            SARI         = (TSMean[i].SARI - TSMean[i - 1].SARI) * sc2 + SARI * sc1;
            Critical     = (TSMean[i].Critical - TSMean[i - 1].Critical) * sc3 + Critical * sc1;
            CritRecov    = (TSMean[i].CritRecov - TSMean[i - 1].CritRecov) * sc4 + CritRecov * sc1;
            incSARI      = TSMean[i].incSARI * (1.0 - sc2) + incSARI * sc1;
            incCritical  = TSMean[i].incCritical * (1.0 - sc3) + incCritical * sc1;
            incCritRecov = TSMean[i].incCritRecov * (1.0 - sc4) + incCritRecov * sc1;
         }
         else
         {
            SARI      = TSMean[i].SARI * sc2;
            Critical  = TSMean[i].Critical * sc3;
            CritRecov = TSMean[i].CritRecov * sc4;
         }

         fprintf(dat,
                 "%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%."
                 "10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%."
                 "10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\n",
                 c * TSMean[i].t, c * TSMean[i].PropSocDist, c * TSMean[i].S, c * TSMean[i].I, c * TSMean[i].R,
                 c * TSMean[i].incI, c * TSMean[i].incC, c * TSMean[i].Mild, c * TSMean[i].ILI, c * TSMean[i].SARI,
                 c * TSMean[i].Critical, c * TSMean[i].CritRecov, c * (TSMean[i].SARI - SARI),
                 c * (TSMean[i].Critical - Critical), c * (TSMean[i].CritRecov - CritRecov), c * TSMean[i].incMild,
                 c * TSMean[i].incILI, c * TSMean[i].incSARI, c * TSMean[i].incCritical, c * TSMean[i].incCritRecov,
                 c * incSARI, c * incCritical, c * incCritRecov, c * TSMean[i].incD, c * TSMean[i].incDeath_ILI,
                 c * TSMean[i].incDeath_SARI, c * TSMean[i].incDeath_Critical, c * TSMean[i].cumMild,
                 c * TSMean[i].cumILI, c * TSMean[i].cumSARI, c * TSMean[i].cumCritical, c * TSMean[i].cumCritRecov,
                 c * TSMean[i].D, c * TSMean[i].cumDeath_ILI, c * TSMean[i].cumDeath_SARI,
                 c * TSMean[i].cumDeath_Critical);
      }
      fclose(dat);

      if ((g_allParams.DoAdUnits) && (g_allParams.OutputSeverityAdminUnit))
      {
         double *SARI_a, *Critical_a, *CritRecov_a, *incSARI_a, *incCritical_a, *incCritRecov_a, sc1a, sc2a, sc3a,
            sc4a; // this stuff corrects bed prevalence for exponentially distributed time to test results in hospital

         if (!(SARI_a = (double*)malloc(MAX_ADUNITS * sizeof(double))))
            ERR_CRITICAL("Unable to allocate temp storage\n");
         if (!(Critical_a = (double*)malloc(MAX_ADUNITS * sizeof(double))))
            ERR_CRITICAL("Unable to allocate temp storage\n");
         if (!(CritRecov_a = (double*)malloc(MAX_ADUNITS * sizeof(double))))
            ERR_CRITICAL("Unable to allocate temp storage\n");
         if (!(incSARI_a = (double*)malloc(MAX_ADUNITS * sizeof(double))))
            ERR_CRITICAL("Unable to allocate temp storage\n");
         if (!(incCritical_a = (double*)malloc(MAX_ADUNITS * sizeof(double))))
            ERR_CRITICAL("Unable to allocate temp storage\n");
         if (!(incCritRecov_a = (double*)malloc(MAX_ADUNITS * sizeof(double))))
            ERR_CRITICAL("Unable to allocate temp storage\n");
         sc1a = (g_allParams.Mean_TimeToTest > 0) ? exp(-1.0 / g_allParams.Mean_TimeToTest) : 0.0;
         sc2a = (g_allParams.Mean_TimeToTest > 0)
                   ? exp(-g_allParams.Mean_TimeToTestOffset / g_allParams.Mean_TimeToTest)
                   : 0.0;
         sc3a = (g_allParams.Mean_TimeToTest > 0)
                   ? exp(-g_allParams.Mean_TimeToTestCriticalOffset / g_allParams.Mean_TimeToTest)
                   : 0.0;
         sc4a = (g_allParams.Mean_TimeToTest > 0)
                   ? exp(-g_allParams.Mean_TimeToTestCritRecovOffset / g_allParams.Mean_TimeToTest)
                   : 0.0;
         for (i = 0; i < g_allParams.NumAdunits; i++)
            incSARI_a[i] = incCritical_a[i] = incCritRecov_a[i] = 0;
         //// output severity results by admin unit
         sprintf(outname, "%s.severity.adunit.xls", OutFilePath);
         if (!(dat = fopen(outname, "wb")))
            ERR_CRITICAL("Unable to open output file\n");
         fprintf(dat, "t");

         /////// ****** /////// ****** /////// ****** COLNAMES
         //// prevalance
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tMild_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tILI_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tSARI_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tCritical_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tCritRecov_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tSARIP_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tCriticalP_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tCritRecovP_%s", AdUnits[i].ad_name);

         //// incidence
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tincI_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tincMild_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tincILI_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tincSARI_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tincCritical_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tincCritRecov_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tincSARIP_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tincCriticalP_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tincCritRecovP_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tincDeath_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tincDeath_ILI_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tincDeath_SARI_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tincDeath__Critical_%s", AdUnits[i].ad_name);

         //// cumulative incidence
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tcumMild_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tcumILI_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tcumSARI_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tcumCritical_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tcumCritRecov_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tcumDeaths_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tcumDeaths_ILI_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tcumDeaths_SARI_%s", AdUnits[i].ad_name);
         for (i = 0; i < g_allParams.NumAdunits; i++)
            fprintf(dat, "\tcumDeaths_Critical_%s", AdUnits[i].ad_name);

         fprintf(dat, "\n");

         /////// ****** /////// ****** /////// ****** Populate table.
         for (i = 0; i < g_allParams.totalSampleNumber; i++)
         {
            for (j = 0; j < g_allParams.NumAdunits; j++)
            {
               if (i > 0)
               {
                  SARI_a[j] = (TSMean[i].SARI_adunit[j] - TSMean[i - 1].SARI_adunit[j]) * sc2a + SARI_a[j] * sc1a;
                  Critical_a[j] =
                     (TSMean[i].Critical_adunit[j] - TSMean[i - 1].Critical_adunit[j]) * sc3a + Critical_a[j] * sc1a;
                  CritRecov_a[j] =
                     (TSMean[i].CritRecov_adunit[j] - TSMean[i - 1].CritRecov_adunit[j]) * sc4a + CritRecov_a[j] * sc1a;
                  incSARI_a[j]      = TSMean[i].incSARI_adunit[j] * (1.0 - sc2a) + incSARI_a[j] * sc1a;
                  incCritical_a[j]  = TSMean[i].incCritical_adunit[j] * (1.0 - sc3a) + incCritical_a[j] * sc1a;
                  incCritRecov_a[j] = TSMean[i].incCritRecov_adunit[j] * (1.0 - sc4a) + incCritRecov_a[j] * sc1a;
               }
               else
               {
                  SARI_a[j]      = TSMean[i].SARI_adunit[j] * sc2a;
                  Critical_a[j]  = TSMean[i].Critical_adunit[j] * sc3a;
                  CritRecov_a[j] = TSMean[i].CritRecov_adunit[j] * sc4a;
               }
            }
            fprintf(dat, "%.10f", c * TSMean[i].t);
            //// prevalance
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", c * TSMean[i].Mild_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", c * TSMean[i].ILI_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", c * TSMean[i].SARI_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", c * TSMean[i].Critical_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", c * TSMean[i].CritRecov_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", c * (TSMean[i].SARI_adunit[j] - SARI_a[j]));
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", c * (TSMean[i].Critical_adunit[j] - Critical_a[j]));
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", c * (TSMean[i].CritRecov_adunit[j] - CritRecov_a[j]));

            //// incidence
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", c * TSMean[i].incI_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", c * TSMean[i].incMild_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", c * TSMean[i].incILI_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", c * TSMean[i].incSARI_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", c * TSMean[i].incCritical_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", c * TSMean[i].incCritRecov_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", c * incSARI_a[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", c * incCritical_a[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", c * incCritRecov_a[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", c * TSMean[i].incD_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", c * TSMean[i].incDeath_ILI_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", c * TSMean[i].incDeath_SARI_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", c * TSMean[i].incDeath_Critical_adunit[j]);

            //// cumulative incidence
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", c * TSMean[i].cumMild_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", c * TSMean[i].cumILI_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", c * TSMean[i].cumSARI_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", c * TSMean[i].cumCritical_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", c * TSMean[i].cumCritRecov_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", c * TSMean[i].cumD_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", c * TSMean[i].cumDeath_ILI_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", c * TSMean[i].cumDeath_SARI_adunit[j]);
            for (j = 0; j < g_allParams.NumAdunits; j++)
               fprintf(dat, "\t%.10f", c * TSMean[i].cumDeath_Critical_adunit[j]);

            if (i != g_allParams.totalSampleNumber - 1)
               fprintf(dat, "\n");
         }
         fclose(dat);
         free(SARI_a);
         free(Critical_a);
         free(CritRecov_a);
         free(incSARI_a);
         free(incCritical_a);
         free(incCritRecov_a);
      }
   }
}

void SaveRandomSeeds(void)
{
   /* function: SaveRandomSeeds(void)
    *
    * Purpose: outputs the random seeds used for each run to a file
    * Parameter: none
    * Returns: none
    *
    * Author: ggilani, 09/03/17
    */
   FILE* dat;
   char outname[1024];

   sprintf(outname, "%s.seeds.xls", OutFilePath);
   if (!(dat = fopen(outname, "wb")))
      ERR_CRITICAL("Unable to open output file\n");
   fprintf(dat, "%li\t%li\n", g_allParams.nextRunSeed1, g_allParams.nextRunSeed2);
   fclose(dat);
}

void SaveEvents(void)
{
   /* function: SaveEvents(void)
    *
    * Purpose: outputs event log to a csv file if required
    * Parameters: none
    * Returns: none
    *
    * Author: ggilani, 15/10/2014
    */
   int i;
   FILE* dat;
   char outname[1024];

   sprintf(outname, "%s.infevents.xls", OutFilePath);
   if (!(dat = fopen(outname, "wb")))
      ERR_CRITICAL("Unable to open output file\n");
   fprintf(dat, "type,t,thread,ind_infectee,cell_infectee,listpos_infectee,adunit_infectee,x_infectee,y_infectee,t_"
                "infector,ind_infector,cell_infector\n");
   for (i = 0; i < *nEvents; i++)
   {
      fprintf(dat, "%i\t%.10f\t%i\t%i\t%i\t%i\t%i\t%.10f\t%.10f\t%.10f\t%i\t%i\n", InfEventLog[i].type,
              InfEventLog[i].t, InfEventLog[i].thread, InfEventLog[i].infectee_ind, InfEventLog[i].infectee_cell,
              InfEventLog[i].listpos, InfEventLog[i].infectee_adunit, InfEventLog[i].infectee_x,
              InfEventLog[i].infectee_y, InfEventLog[i].t_infector, InfEventLog[i].infector_ind,
              InfEventLog[i].infector_cell);
   }
   fclose(dat);
}

void LoadSnapshot(void)
{
   FILE* dat;
   int i, j, *CellMemberArray, *CellSuscMemberArray;
   long l;
   long long CM_offset, CSM_offset;
   double t;
   int** Array_InvCDF;
   float *Array_tot_prob, **Array_cum_trans, **Array_max_trans;

   if (!(dat = fopen(SnapshotLoadFile, "rb")))
      ERR_CRITICAL("Unable to open snapshot file\n");
   fprintf(stderr, "Loading snapshot.");
   if (!(Array_InvCDF = (int**)malloc(g_allParams.populatedCellCount * sizeof(int*))))
      ERR_CRITICAL("Unable to allocate temp cell storage\n");
   if (!(Array_max_trans = (float**)malloc(g_allParams.populatedCellCount * sizeof(float*))))
      ERR_CRITICAL("Unable to temp allocate cell storage\n");
   if (!(Array_cum_trans = (float**)malloc(g_allParams.populatedCellCount * sizeof(float*))))
      ERR_CRITICAL("Unable to temp allocate cell storage\n");
   if (!(Array_tot_prob = (float*)malloc(g_allParams.populatedCellCount * sizeof(float))))
      ERR_CRITICAL("Unable to temp allocate cell storage\n");
   for (i = 0; i < g_allParams.populatedCellCount; i++)
   {
      Array_InvCDF[i]    = Cells[i].InvCDF;
      Array_max_trans[i] = Cells[i].max_trans;
      Array_cum_trans[i] = Cells[i].cum_trans;
      Array_tot_prob[i]  = Cells[i].tot_prob;
   }

   fread_big((void*)&i, sizeof(int), 1, dat);
   if (i != g_allParams.populationSize)
      ERR_CRITICAL_FMT("Incorrect N (%i %i) in snapshot file.\n", g_allParams.populationSize, i);
   fread_big((void*)&i, sizeof(int), 1, dat);
   if (i != g_allParams.housholdCount)
      ERR_CRITICAL("Incorrect NH in snapshot file.\n");
   fread_big((void*)&i, sizeof(int), 1, dat);
   if (i != g_allParams.cellCount)
      ERR_CRITICAL_FMT("## %i neq %i\nIncorrect NC in snapshot file.", i, g_allParams.cellCount);
   fread_big((void*)&i, sizeof(int), 1, dat);
   if (i != g_allParams.populatedCellCount)
      ERR_CRITICAL("Incorrect NCP in snapshot file.\n");
   fread_big((void*)&i, sizeof(int), 1, dat);
   if (i != g_allParams.ncw)
      ERR_CRITICAL("Incorrect ncw in snapshot file.\n");
   fread_big((void*)&i, sizeof(int), 1, dat);
   if (i != g_allParams.nch)
      ERR_CRITICAL("Incorrect nch in snapshot file.\n");
   fread_big((void*)&l, sizeof(long), 1, dat);
   if (l != g_allParams.setupSeed1)
      ERR_CRITICAL("Incorrect setupSeed1 in snapshot file.\n");
   fread_big((void*)&l, sizeof(long), 1, dat);
   if (l != g_allParams.setupSeed2)
      ERR_CRITICAL("Incorrect setupSeed2 in snapshot file.\n");
   fread_big((void*)&t, sizeof(double), 1, dat);
   if (t != g_allParams.TimeStep)
      ERR_CRITICAL("Incorrect TimeStep in snapshot file.\n");
   fread_big((void*)&(g_allParams.SnapshotLoadTime), sizeof(double), 1, dat);
   g_allParams.totalSampleNumber =
      1 + (int)ceil((g_allParams.SampleTime - g_allParams.SnapshotLoadTime) / g_allParams.SampleStep);
   fprintf(stderr, ".");
   fread_big((void*)&CellMemberArray, sizeof(int*), 1, dat);
   fprintf(stderr, ".");
   fread_big((void*)&CellSuscMemberArray, sizeof(int*), 1, dat);
   fprintf(stderr, ".");
   CM_offset  = State.CellMemberArray - CellMemberArray;
   CSM_offset = State.CellSuscMemberArray - CellSuscMemberArray;

   zfread_big((void*)Hosts, sizeof(person), (size_t)g_allParams.populationSize, dat);
   fprintf(stderr, ".");
   zfread_big((void*)Households, sizeof(household), (size_t)g_allParams.housholdCount, dat);
   fprintf(stderr, ".");
   zfread_big((void*)Cells, sizeof(cell), (size_t)g_allParams.cellCount, dat);
   fprintf(stderr, ".");
   zfread_big((void*)Mcells, sizeof(microcell), (size_t)g_allParams.microcellCount, dat);
   fprintf(stderr, ".");
   zfread_big((void*)State.CellMemberArray, sizeof(int), (size_t)g_allParams.populationSize, dat);
   fprintf(stderr, ".");
   zfread_big((void*)State.CellSuscMemberArray, sizeof(int), (size_t)g_allParams.populationSize, dat);
   fprintf(stderr, ".");
   for (i = 0; i < g_allParams.cellCount; i++)
   {
      if (Cells[i].n > 0)
      {
         Cells[i].members += CM_offset;
         Cells[i].susceptible += CSM_offset;
         Cells[i].latent += CSM_offset;
         Cells[i].infected += CSM_offset;
      }
      for (j = 0; j < MAX_INTERVENTION_TYPES; j++)
         Cells[i].CurInterv[j] = -1; // turn interventions off in loaded image
   }
   for (i = 0; i < g_allParams.microcellCount; i++)
      if (Mcells[i].n > 0)
         Mcells[i].members += CM_offset;

   for (i = 0; i < g_allParams.populatedCellCount; i++)
   {
      Cells[i].InvCDF    = Array_InvCDF[i];
      Cells[i].max_trans = Array_max_trans[i];
      Cells[i].cum_trans = Array_cum_trans[i];
      Cells[i].tot_prob  = Array_tot_prob[i];
   }
   free(Array_tot_prob);
   free(Array_cum_trans);
   free(Array_max_trans);
   free(Array_InvCDF);
   fprintf(stderr, "\n");
   fclose(dat);
}

void SaveSnapshot(void)
{
   FILE* dat;
   int i = 1;

   if (!(dat = fopen(SnapshotSaveFile, "wb")))
      ERR_CRITICAL("Unable to open snapshot file\n");

   fwrite_big((void*)&(g_allParams.populationSize), sizeof(int), 1, dat);
   fprintf(stderr, "## %i\n", i++);
   fwrite_big((void*)&(g_allParams.housholdCount), sizeof(int), 1, dat);
   fprintf(stderr, "## %i\n", i++);
   fwrite_big((void*)&(g_allParams.cellCount), sizeof(int), 1, dat);
   fprintf(stderr, "## %i\n", i++);
   fwrite_big((void*)&(g_allParams.populatedCellCount), sizeof(int), 1, dat);
   fprintf(stderr, "## %i\n", i++);
   fwrite_big((void*)&(g_allParams.ncw), sizeof(int), 1, dat);
   fprintf(stderr, "## %i\n", i++);
   fwrite_big((void*)&(g_allParams.nch), sizeof(int), 1, dat);
   fprintf(stderr, "## %i\n", i++);
   fwrite_big((void*)&(g_allParams.setupSeed1), sizeof(long), 1, dat);
   fprintf(stderr, "## %i\n", i++);
   fwrite_big((void*)&(g_allParams.setupSeed2), sizeof(long), 1, dat);
   fprintf(stderr, "## %i\n", i++);
   fwrite_big((void*)&(g_allParams.TimeStep), sizeof(double), 1, dat);
   fprintf(stderr, "## %i\n", i++);
   fwrite_big((void*)&(g_allParams.SnapshotSaveTime), sizeof(double), 1, dat);
   fprintf(stderr, "## %i\n", i++);
   fwrite_big((void*)&(State.CellMemberArray), sizeof(int*), 1, dat);
   fprintf(stderr, "## %i\n", i++);
   fwrite_big((void*)&(State.CellSuscMemberArray), sizeof(int*), 1, dat);
   fprintf(stderr, "## %i\n", i++);

   zfwrite_big((void*)Hosts, sizeof(person), (size_t)g_allParams.populationSize, dat);

   fprintf(stderr, "## %i\n", i++);
   zfwrite_big((void*)Households, sizeof(household), (size_t)g_allParams.housholdCount, dat);
   fprintf(stderr, "## %i\n", i++);
   zfwrite_big((void*)Cells, sizeof(cell), (size_t)g_allParams.cellCount, dat);
   fprintf(stderr, "## %i\n", i++);
   zfwrite_big((void*)Mcells, sizeof(microcell), (size_t)g_allParams.microcellCount, dat);
   fprintf(stderr, "## %i\n", i++);

   zfwrite_big((void*)State.CellMemberArray, sizeof(int), (size_t)g_allParams.populationSize, dat);
   fprintf(stderr, "## %i\n", i++);
   zfwrite_big((void*)State.CellSuscMemberArray, sizeof(int), (size_t)g_allParams.populationSize, dat);
   fprintf(stderr, "## %i\n", i++);

   fclose(dat);
}

void UpdateProbs(int DoPlace)
{
   int j;

   if (!DoPlace)
   {
#pragma omp parallel for private(j) schedule(static, 500)
      for (j = 0; j < g_allParams.populatedCellCount; j++)
      {
         CellLookup[j]->tot_prob = 0;
         CellLookup[j]->S0       = CellLookup[j]->S + CellLookup[j]->L + CellLookup[j]->I;
         if (g_allParams.DoDeath)
         {
            CellLookup[j]->S0 += CellLookup[j]->n / 5;
            if ((CellLookup[j]->n < 100) || (CellLookup[j]->S0 > CellLookup[j]->n))
               CellLookup[j]->S0 = CellLookup[j]->n;
         }
      }
   }
   else
   {
#pragma omp parallel for private(j) schedule(static, 500)
      for (j = 0; j < g_allParams.populatedCellCount; j++)
      {
         CellLookup[j]->S0       = CellLookup[j]->S;
         CellLookup[j]->tot_prob = 0;
      }
   }
#pragma omp parallel for private(j) schedule(static, 500)
   for (j = 0; j < g_allParams.populatedCellCount; j++)
   {
      int m, k;
      float t;
      CellLookup[j]->cum_trans[0] = ((float)(CellLookup[0]->S0)) * CellLookup[j]->max_trans[0];
      t                           = ((float)CellLookup[0]->n) * CellLookup[j]->max_trans[0];
      for (m = 1; m < g_allParams.populatedCellCount; m++)
      {
         CellLookup[j]->cum_trans[m] =
            CellLookup[j]->cum_trans[m - 1] + ((float)(CellLookup[m]->S0)) * CellLookup[j]->max_trans[m];
         t += ((float)CellLookup[m]->n) * CellLookup[j]->max_trans[m];
      }
      CellLookup[j]->tot_prob = CellLookup[j]->cum_trans[g_allParams.populatedCellCount - 1];
      for (m = 0; m < g_allParams.populatedCellCount; m++)
         CellLookup[j]->cum_trans[m] /= CellLookup[j]->tot_prob;
      CellLookup[j]->tot_prob /= t;
      for (k = m = 0; k <= 1024; k++)
      {
         while (CellLookup[j]->cum_trans[m] * 1024 < ((float)k))
            m++;
         CellLookup[j]->InvCDF[k] = m;
      }
   }
}

int ChooseTriggerVariableAndValue(int AdUnit)
{
   int VariableAndValue = 0;
   if (g_allParams.DoGlobalTriggers)
   {
      if (g_allParams.DoPerCapitaTriggers)
         VariableAndValue =
            (int)floor(((double)State.trigDC) * g_allParams.GlobalIncThreshPop / ((double)g_allParams.populationSize));
      else
         VariableAndValue = State.trigDC;
   }
   else if (g_allParams.DoAdminTriggers)
      VariableAndValue = State.trigDC_adunit[AdUnit];
   else
      VariableAndValue = INT_MAX; //// i.e. if not doing triggering (at either admin or global level) then set value to
                                  ///be arbitrarily large so that it will surpass any trigger threshold. Probably other
                                  ///ways around this if anybody wants to correct?

   return VariableAndValue;
}

double ChooseThreshold(
   int AdUnit,
   double WhichThreshold) //// point is that this threshold needs to be generalised, so this is likely insufficient.
{
   double Threshold = 0;
   if (g_allParams.DoGlobalTriggers)
      Threshold = WhichThreshold;
   else if (g_allParams.DoAdminTriggers)
   {
      if (g_allParams.DoPerCapitaTriggers)
         Threshold = (int)ceil(((double)(AdUnits[AdUnit].n * WhichThreshold)) / g_allParams.IncThreshPop);
      else
         Threshold = WhichThreshold;
   }
   return Threshold;
}

void DoOrDontAmendStartTime(double* StartTimeToAmend, double StartTime)
{
   if (*StartTimeToAmend >= 1e10)
      *StartTimeToAmend = StartTime;
}

void UpdateEfficaciesAndComplianceProportions(double t)
{
   //// **** social distancing
   for (int ChangeTime = 0; ChangeTime < g_allParams.Num_SD_ChangeTimes; ChangeTime++)
      if (t == g_allParams.SD_ChangeTimes[ChangeTime])
      {
         //// **** non-enhanced
         g_allParams.SocDistHouseholdEffectCurrent =
            g_allParams.SD_HouseholdEffects_OverTime[ChangeTime];                                      //// household
         g_allParams.SocDistSpatialEffectCurrent = g_allParams.SD_SpatialEffects_OverTime[ChangeTime]; //// spatial
         for (int PlaceType = 0; PlaceType < g_allParams.PlaceTypeNum; PlaceType++)
            g_allParams.SocDistPlaceEffectCurrent[PlaceType] =
               g_allParams.SD_PlaceEffects_OverTime[ChangeTime][PlaceType]; ///// place

         //// **** enhanced
         g_allParams.EnhancedSocDistHouseholdEffectCurrent =
            g_allParams.Enhanced_SD_HouseholdEffects_OverTime[ChangeTime]; //// household
         g_allParams.EnhancedSocDistSpatialEffectCurrent =
            g_allParams.Enhanced_SD_SpatialEffects_OverTime[ChangeTime]; //// spatial
         for (int PlaceType = 0; PlaceType < g_allParams.PlaceTypeNum; PlaceType++)
            g_allParams.EnhancedSocDistPlaceEffectCurrent[PlaceType] =
               g_allParams.Enhanced_SD_PlaceEffects_OverTime[ChangeTime][PlaceType]; ///// place

         g_allParams.SocDistCellIncThresh =
            g_allParams.SD_CellIncThresh_OverTime[ChangeTime]; //// cell incidence threshold
      }

   //// **** case isolation
   for (int ChangeTime = 0; ChangeTime < g_allParams.Num_CI_ChangeTimes; ChangeTime++)
      if (t == g_allParams.CI_ChangeTimes[ChangeTime])
      {
         g_allParams.CaseIsolationEffectiveness =
            g_allParams.CI_SpatialAndPlaceEffects_OverTime[ChangeTime]; //// spatial / place
         g_allParams.CaseIsolationHouseEffectiveness =
            g_allParams.CI_HouseholdEffects_OverTime[ChangeTime]; //// household

         g_allParams.CaseIsolationProp = g_allParams.CI_Prop_OverTime[ChangeTime]; //// compliance
         g_allParams.CaseIsolation_CellIncThresh =
            g_allParams.CI_CellIncThresh_OverTime[ChangeTime]; //// cell incidence threshold
      }

   ////// **** household quarantine
   if (g_allParams.DoHouseholds)
      for (int ChangeTime = 0; ChangeTime < g_allParams.Num_HQ_ChangeTimes; ChangeTime++)
         if (t == g_allParams.HQ_ChangeTimes[ChangeTime])
         {
            g_allParams.HQuarantineSpatialEffect = g_allParams.HQ_SpatialEffects_OverTime[ChangeTime];   //// spatial
            g_allParams.HQuarantineHouseEffect   = g_allParams.HQ_HouseholdEffects_OverTime[ChangeTime]; //// household
            for (int PlaceType = 0; PlaceType < g_allParams.PlaceTypeNum; PlaceType++)
               g_allParams.HQuarantinePlaceEffect[PlaceType] =
                  g_allParams.HQ_PlaceEffects_OverTime[ChangeTime][PlaceType]; //// place

            g_allParams.HQuarantinePropIndivCompliant =
               g_allParams.HQ_Individual_PropComply_OverTime[ChangeTime]; //// individual compliance
            g_allParams.HQuarantinePropHouseCompliant =
               g_allParams.HQ_Household_PropComply_OverTime[ChangeTime]; //// household compliance

            g_allParams.HHQuar_CellIncThresh =
               g_allParams.HQ_CellIncThresh_OverTime[ChangeTime]; //// cell incidence threshold
         }

   //// **** place closure
   if (g_allParams.DoPlaces)
   {
      for (int ChangeTime = 0; ChangeTime < g_allParams.Num_PC_ChangeTimes; ChangeTime++)
         if (t == g_allParams.PC_ChangeTimes[ChangeTime])
         {
            //// First open all the places - keep commented out in case becomes necessary but avoid if possible to avoid
            ///runtime costs.
            //				unsigned short int ts = (unsigned short int) (g_allParams.TimeStepsPerDay * t);
            //				for (int PlaceType = 0; PlaceType < g_allParams.PlaceTypeNum; PlaceType++)
            //#pragma omp parallel for schedule(static,1)
            //					for (int ThreadNum = 0; ThreadNum < g_allParams.NumThreads; ThreadNum++)
            //						for (int PlaceNum = ThreadNum; PlaceNum < g_allParams.Nplace[PlaceType]; PlaceNum +=
            //g_allParams.NumThreads) 							DoPlaceOpen(PlaceType, PlaceNum, ts, ThreadNum);

            g_allParams.PlaceCloseSpatialRelContact = g_allParams.PC_SpatialEffects_OverTime[ChangeTime]; //// spatial
            g_allParams.PlaceCloseHouseholdRelContact =
               g_allParams.PC_HouseholdEffects_OverTime[ChangeTime]; //// household
            for (int PlaceType = 0; PlaceType < g_allParams.PlaceTypeNum; PlaceType++)
            {
               g_allParams.PlaceCloseEffect[PlaceType] =
                  g_allParams.PC_PlaceEffects_OverTime[ChangeTime][PlaceType]; //// place
               g_allParams.PlaceClosePropAttending[PlaceType] =
                  g_allParams.PC_PropAttending_OverTime[ChangeTime][PlaceType]; //// place
            }

            g_allParams.PlaceCloseIncTrig =
               g_allParams.PC_IncThresh_OverTime[ChangeTime]; //// global incidence threshold
            g_allParams.PlaceCloseFracIncTrig =
               g_allParams.PC_FracIncThresh_OverTime[ChangeTime]; //// fractional incidence threshold
            g_allParams.PlaceCloseCellIncThresh =
               g_allParams.PC_CellIncThresh_OverTime[ChangeTime];                      //// cell incidence threshold
            g_allParams.PlaceCloseDuration = g_allParams.PC_Durs_OverTime[ChangeTime]; //// duration of place closure

            //// reset place close time start - has been set to 9e9 in event of no triggers. m
            if (g_allParams.PlaceCloseTimeStart < 1e10)
               g_allParams.PlaceCloseTimeStart = t;

            // ensure that new duration doesn't go over next change time. Judgement call here - talk to Neil if this is
            // what he wants.
            if ((ChangeTime < g_allParams.Num_PC_ChangeTimes - 1)
                && (g_allParams.PlaceCloseTimeStart + g_allParams.PlaceCloseDuration
                    >= g_allParams.PC_ChangeTimes[ChangeTime + 1]))
               g_allParams.PlaceCloseDuration =
                  g_allParams.PC_ChangeTimes[ChangeTime + 1] - g_allParams.PC_ChangeTimes[ChangeTime] - 1;
            // fprintf(stderr, "\nt=%lf, n=%i (%i)  PlaceCloseDuration = %lf  (%lf) \n", t, ChangeTime,
            // g_allParams.Num_PC_ChangeTimes, g_allParams.PlaceCloseDuration, g_allParams.PC_ChangeTimes[ChangeTime+1]);
         }
   }

   //// **** digital contact tracing
   for (int ChangeTime = 0; ChangeTime < g_allParams.Num_DCT_ChangeTimes; ChangeTime++)
      if (t == g_allParams.DCT_ChangeTimes[ChangeTime])
      {
         g_allParams.DCTCaseIsolationEffectiveness =
            g_allParams.DCT_SpatialAndPlaceEffects_OverTime[ChangeTime]; //// spatial / place
         g_allParams.DCTCaseIsolationHouseEffectiveness =
            g_allParams.DCT_HouseholdEffects_OverTime[ChangeTime];                                 //// household
         g_allParams.ProportionDigitalContactsIsolate = g_allParams.DCT_Prop_OverTime[ChangeTime]; //// compliance
         g_allParams.MaxDigitalContactsToTrace        = g_allParams.DCT_MaxToTrace_OverTime[ChangeTime];
      }
}

void RecordSample(double t, int n)
{
   int i, j, k, S, L, I, R, D, N, cumC, cumTC, cumI, cumR, cumD, cumDC, cumFC;
   int cumH;   // add number of hospitalised, cumulative hospitalisation: ggilani 28/10/14
   int cumCT;  // added cumulative number of contact traced: ggilani 15/06/17
   int cumCC;  // added cumulative number of cases who are contacts: ggilani 28/05/2019
   int cumDCT; // added cumulative number of cases who are digitally contact traced: ggilani 11/03/20
   int cumHQ, cumAC, cumAH, cumAA, cumACS, cumAPC, cumAPA, cumAPCS, numPC, trigDC, trigAlert, trigAlertC;
   int cumC_country[MAX_COUNTRIES]; // add cumulative cases per country
   cell* ct;
   unsigned short int ts;
   double s, thr;

   //// Severity quantities
   int Mild, ILI, SARI, Critical, CritRecov, cumMild, cumILI, cumSARI, cumCritical, cumCritRecov, cumDeath_ILI,
      cumDeath_SARI, cumDeath_Critical;

   ts = (unsigned short int)(g_allParams.TimeStepsPerDay * t);

   //// initialize to zero
   S = L = I = R = D = cumI = cumC = cumDC = cumTC = cumFC = cumHQ = cumAC = cumAA = cumAH = cumACS = cumAPC = cumAPA =
      cumAPCS = cumD = cumH = cumCT = cumCC = cumDCT = 0;
   for (i = 0; i < MAX_COUNTRIES; i++)
      cumC_country[i] = 0;
   if (g_allParams.DoSeverity)
      Mild = ILI = SARI = Critical = CritRecov = cumMild = cumILI = cumSARI = cumCritical = cumCritRecov =
         cumDeath_ILI = cumDeath_SARI = cumDeath_Critical = 0;

#pragma omp parallel for private(i, ct) schedule(static, 10000) reduction(+ : S, L, I, R, D, cumTC) // added i to
                                                                                                    // private
   for (i = 0; i < g_allParams.populatedCellCount; i++)
   {
      ct = CellLookup[i];
      S += (int)ct->S;
      L += (int)ct->L;
      I += (int)ct->I;
      R += (int)ct->R;
      D += (int)ct->D;
      cumTC += (int)ct->cumTC;
   }
   cumR = R;
   cumD = D;
   // cumD = 0;
   N = S + L + I + R + D;
   if (N != g_allParams.populationSize)
      fprintf(stderr, "## %i #\n", g_allParams.populationSize - N);
   State.sumRad2 = 0;
   for (j = 0; j < g_allParams.NumThreads; j++)
   {
      cumI += StateT[j].cumI;
      cumC += StateT[j].cumC;
      cumDC += StateT[j].cumDC;
      cumFC += StateT[j].cumFC;
      cumH += StateT[j].cumH;     // added cumulative hospitalisation
      cumCT += StateT[j].cumCT;   // added contact tracing
      cumCC += StateT[j].cumCC;   // added cases who are contacts
      cumDCT += StateT[j].cumDCT; // added cases who are digitally contact traced
      State.sumRad2 += StateT[j].sumRad2;
      State.sumRad2 += StateT[j].sumRad2;
      cumHQ += StateT[j].cumHQ;
      cumAC += StateT[j].cumAC;
      cumAA += StateT[j].cumAA;
      cumAPC += StateT[j].cumAPC;
      cumAPA += StateT[j].cumAPA;
      cumAPCS += StateT[j].cumAPCS;
      cumAH += StateT[j].cumAH;
      cumACS += StateT[j].cumACS;
      // cumD += StateT[j].cumD;

      if (g_allParams.DoSeverity)
      {
         ///// severity states by thread
         Mild += StateT[j].Mild;
         ILI += StateT[j].ILI;
         SARI += StateT[j].SARI;
         Critical += StateT[j].Critical;
         CritRecov += StateT[j].CritRecov;

         ///// cumulative severity states by thread
         cumMild += StateT[j].cumMild;
         cumILI += StateT[j].cumILI;
         cumSARI += StateT[j].cumSARI;
         cumCritical += StateT[j].cumCritical;
         cumCritRecov += StateT[j].cumCritRecov;
         cumDeath_ILI += StateT[j].cumDeath_ILI;
         cumDeath_SARI += StateT[j].cumDeath_SARI;
         cumDeath_Critical += StateT[j].cumDeath_Critical;
      }

      // add up cumulative country counts: ggilani - 12/11/14
      for (i = 0; i < MAX_COUNTRIES; i++)
         cumC_country[i] += StateT[j].cumC_country[i];
      if (State.maxRad2 < StateT[j].maxRad2)
         State.maxRad2 = StateT[j].maxRad2;
   }
   for (j = 0; j < g_allParams.NumThreads; j++)
      StateT[j].maxRad2 = State.maxRad2;
   TimeSeries[n].t     = t;
   TimeSeries[n].S     = (double)S;
   TimeSeries[n].L     = (double)L;
   TimeSeries[n].I     = (double)I;
   TimeSeries[n].R     = (double)R;
   TimeSeries[n].D     = (double)D;
   TimeSeries[n].incI  = (double)(cumI - State.cumI);
   TimeSeries[n].incC  = (double)(cumC - State.cumC);
   TimeSeries[n].incFC = (double)(cumFC - State.cumFC);
   TimeSeries[n].incH = (double)(cumH - State.cumH);       // added incidence of hospitalisation
   TimeSeries[n].incCT  = (double)(cumCT - State.cumCT);   // added contact tracing
   TimeSeries[n].incCC  = (double)(cumCC - State.cumCC);   // added cases who are contacts
   TimeSeries[n].incDCT = (double)(cumDCT - State.cumDCT); // added cases who are digitally contact traced
   TimeSeries[n].incDC = (double)(cumDC - State.cumDC);    // added incidence of detected cases
   TimeSeries[n].incTC   = (double)(cumTC - State.cumTC);
   TimeSeries[n].incR    = (double)(cumR - State.cumR);
   TimeSeries[n].incD    = (double)(cumD - State.cumD);
   TimeSeries[n].incHQ   = (double)(cumHQ - State.cumHQ);
   TimeSeries[n].incAC   = (double)(cumAC - State.cumAC);
   TimeSeries[n].incAH   = (double)(cumAH - State.cumAH);
   TimeSeries[n].incAA   = (double)(cumAA - State.cumAA);
   TimeSeries[n].incACS  = (double)(cumACS - State.cumACS);
   TimeSeries[n].incAPC  = (double)(cumAPC - State.cumAPC);
   TimeSeries[n].incAPA  = (double)(cumAPA - State.cumAPA);
   TimeSeries[n].incAPCS = (double)(cumAPCS - State.cumAPCS);
   TimeSeries[n].cumT    = State.cumT;
   TimeSeries[n].cumUT   = State.cumUT;
   TimeSeries[n].cumTP   = State.cumTP;
   TimeSeries[n].cumV    = State.cumV;
   TimeSeries[n].cumVG   = State.cumVG; // added VG;
   TimeSeries[n].cumDC   = cumDC;
   // fprintf(stderr, "\ncumD=%i last_cumD=%i incD=%lg\n ", cumD, State.cumD, TimeSeries[n].incD);
   // incidence per country
   for (i = 0; i < MAX_COUNTRIES; i++)
      TimeSeries[n].incC_country[i] = (double)(cumC_country[i] - State.cumC_country[i]);
   if (g_allParams.DoICUTriggers)
   {
      trigDC = cumCritical;
      if (n >= g_allParams.TriggersSamplingInterval)
         trigDC -= (int)TimeSeries[n - g_allParams.TriggersSamplingInterval].cumCritical;
   }
   else
   {
      trigDC = cumDC;
      if (n >= g_allParams.TriggersSamplingInterval)
         trigDC -= (int)TimeSeries[n - g_allParams.TriggersSamplingInterval].cumDC;
   }
   State.trigDC = trigDC;

   //// update State with new totals from threads.
   State.S       = S;
   State.L       = L;
   State.I       = I;
   State.R       = R;
   State.D       = D;
   State.cumI    = cumI;
   State.cumDC   = cumDC;
   State.cumTC   = cumTC;
   State.cumFC   = cumFC;
   State.cumH    = cumH;   // added cumulative hospitalisation
   State.cumCT   = cumCT;  // added cumulative contact tracing
   State.cumCC   = cumCC;  // added cumulative cases who are contacts
   State.cumDCT  = cumDCT; // added cumulative cases who are digitally contact traced
   State.cumC    = cumC;
   State.cumR    = cumR;
   State.cumD    = cumD;
   State.cumHQ   = cumHQ;
   State.cumAC   = cumAC;
   State.cumAH   = cumAH;
   State.cumAA   = cumAA;
   State.cumACS  = cumACS;
   State.cumAPC  = cumAPC;
   State.cumAPA  = cumAPA;
   State.cumAPCS = cumAPCS;

   if (g_allParams.DoSeverity)
   {
      //// Record incidence. (Must be done with old State totals)
      TimeSeries[n].incMild           = (double)(cumMild - State.cumMild);
      TimeSeries[n].incILI            = (double)(cumILI - State.cumILI);
      TimeSeries[n].incSARI           = (double)(cumSARI - State.cumSARI);
      TimeSeries[n].incCritical       = (double)(cumCritical - State.cumCritical);
      TimeSeries[n].incCritRecov      = (double)(cumCritRecov - State.cumCritRecov);
      TimeSeries[n].incDeath_ILI      = (double)(cumDeath_ILI - State.cumDeath_ILI);
      TimeSeries[n].incDeath_SARI     = (double)(cumDeath_SARI - State.cumDeath_SARI);
      TimeSeries[n].incDeath_Critical = (double)(cumDeath_Critical - State.cumDeath_Critical);

      /////// update state with totals
      State.Mild              = Mild;
      State.ILI               = ILI;
      State.SARI              = SARI;
      State.Critical          = Critical;
      State.CritRecov         = CritRecov;
      State.cumMild           = cumMild;
      State.cumILI            = cumILI;
      State.cumSARI           = cumSARI;
      State.cumCritical       = cumCritical;
      State.cumCritRecov      = cumCritRecov;
      State.cumDeath_ILI      = cumDeath_ILI;
      State.cumDeath_SARI     = cumDeath_SARI;
      State.cumDeath_Critical = cumDeath_Critical;

      //// Record new totals for time series. (Must be done with old State totals)
      TimeSeries[n].Mild              = Mild;
      TimeSeries[n].ILI               = ILI;
      TimeSeries[n].SARI              = SARI;
      TimeSeries[n].Critical          = Critical;
      TimeSeries[n].CritRecov         = CritRecov;
      TimeSeries[n].cumMild           = cumMild;
      TimeSeries[n].cumILI            = cumILI;
      TimeSeries[n].cumSARI           = cumSARI;
      TimeSeries[n].cumCritical       = cumCritical;
      TimeSeries[n].cumCritRecov      = cumCritRecov;
      TimeSeries[n].cumDeath_ILI      = cumDeath_ILI;
      TimeSeries[n].cumDeath_SARI     = cumDeath_SARI;
      TimeSeries[n].cumDeath_Critical = cumDeath_Critical;

      if (g_allParams.DoAdUnits)
         for (i = 0; i <= g_allParams.NumAdunits; i++)
         {
            //// Record incidence. Need new total minus old total (same as minus old total plus new total).
            //// First subtract old total while unchanged.
            TimeSeries[n].incMild_adunit[i]           = (double)(-State.cumMild_adunit[i]);
            TimeSeries[n].incILI_adunit[i]            = (double)(-State.cumILI_adunit[i]);
            TimeSeries[n].incSARI_adunit[i]           = (double)(-State.cumSARI_adunit[i]);
            TimeSeries[n].incCritical_adunit[i]       = (double)(-State.cumCritical_adunit[i]);
            TimeSeries[n].incCritRecov_adunit[i]      = (double)(-State.cumCritRecov_adunit[i]);
            TimeSeries[n].incD_adunit[i]              = (double)(-State.cumD_adunit[i]);
            TimeSeries[n].incDeath_ILI_adunit[i]      = (double)(-State.cumDeath_ILI_adunit[i]);
            TimeSeries[n].incDeath_SARI_adunit[i]     = (double)(-State.cumDeath_SARI_adunit[i]);
            TimeSeries[n].incDeath_Critical_adunit[i] = (double)(-State.cumDeath_Critical_adunit[i]);

            //// reset State (don't think StateT) to zero. Don't need to do this with non-admin unit as local variables
            ///Mild, cumSARI etc. initialized to zero at beginning of function. Check with Gemma
            State.Mild_adunit[i]              = 0;
            State.ILI_adunit[i]               = 0;
            State.SARI_adunit[i]              = 0;
            State.Critical_adunit[i]          = 0;
            State.CritRecov_adunit[i]         = 0;
            State.cumMild_adunit[i]           = 0;
            State.cumILI_adunit[i]            = 0;
            State.cumSARI_adunit[i]           = 0;
            State.cumCritical_adunit[i]       = 0;
            State.cumCritRecov_adunit[i]      = 0;
            State.cumD_adunit[i]              = 0;
            State.cumDeath_ILI_adunit[i]      = 0;
            State.cumDeath_SARI_adunit[i]     = 0;
            State.cumDeath_Critical_adunit[i] = 0;

            for (j = 0; j < g_allParams.NumThreads; j++)
            {
               //// collate from threads
               State.Mild_adunit[i] += StateT[j].Mild_adunit[i];
               State.ILI_adunit[i] += StateT[j].ILI_adunit[i];
               State.SARI_adunit[i] += StateT[j].SARI_adunit[i];
               State.Critical_adunit[i] += StateT[j].Critical_adunit[i];
               State.CritRecov_adunit[i] += StateT[j].CritRecov_adunit[i];
               State.cumMild_adunit[i] += StateT[j].cumMild_adunit[i];
               State.cumILI_adunit[i] += StateT[j].cumILI_adunit[i];
               State.cumSARI_adunit[i] += StateT[j].cumSARI_adunit[i];
               State.cumCritical_adunit[i] += StateT[j].cumCritical_adunit[i];
               State.cumCritRecov_adunit[i] += StateT[j].cumCritRecov_adunit[i];
               State.cumD_adunit[i] += StateT[j].cumD_adunit[i];
               State.cumDeath_ILI_adunit[i] += StateT[j].cumDeath_ILI_adunit[i];
               State.cumDeath_SARI_adunit[i] += StateT[j].cumDeath_SARI_adunit[i];
               State.cumDeath_Critical_adunit[i] += StateT[j].cumDeath_Critical_adunit[i];
            }

            //// Record incidence. Need new total minus old total. Add new total
            TimeSeries[n].incMild_adunit[i] += (double)(State.cumMild_adunit[i]);
            TimeSeries[n].incILI_adunit[i] += (double)(State.cumILI_adunit[i]);
            TimeSeries[n].incSARI_adunit[i] += (double)(State.cumSARI_adunit[i]);
            TimeSeries[n].incCritical_adunit[i] += (double)(State.cumCritical_adunit[i]);
            TimeSeries[n].incCritRecov_adunit[i] += (double)(State.cumCritRecov_adunit[i]);
            TimeSeries[n].incD_adunit[i] += (double)(State.cumD_adunit[i]);
            TimeSeries[n].incDeath_ILI_adunit[i] += (double)(State.cumDeath_ILI_adunit[i]);
            TimeSeries[n].incDeath_SARI_adunit[i] += (double)(State.cumDeath_SARI_adunit[i]);
            TimeSeries[n].incDeath_Critical_adunit[i] += (double)(State.cumDeath_Critical_adunit[i]);

            //// Record new totals for time series. (Must be done with old State totals)
            TimeSeries[n].Mild_adunit[i]              = State.Mild_adunit[i];
            TimeSeries[n].ILI_adunit[i]               = State.ILI_adunit[i];
            TimeSeries[n].SARI_adunit[i]              = State.SARI_adunit[i];
            TimeSeries[n].Critical_adunit[i]          = State.Critical_adunit[i];
            TimeSeries[n].CritRecov_adunit[i]         = State.CritRecov_adunit[i];
            TimeSeries[n].cumMild_adunit[i]           = State.cumMild_adunit[i];
            TimeSeries[n].cumILI_adunit[i]            = State.cumILI_adunit[i];
            TimeSeries[n].cumSARI_adunit[i]           = State.cumSARI_adunit[i];
            TimeSeries[n].cumCritical_adunit[i]       = State.cumCritical_adunit[i];
            TimeSeries[n].cumCritRecov_adunit[i]      = State.cumCritRecov_adunit[i];
            TimeSeries[n].cumD_adunit[i]              = State.cumD_adunit[i];
            TimeSeries[n].cumDeath_ILI_adunit[i]      = State.cumDeath_ILI_adunit[i];
            TimeSeries[n].cumDeath_SARI_adunit[i]     = State.cumDeath_SARI_adunit[i];
            TimeSeries[n].cumDeath_Critical_adunit[i] = State.cumDeath_Critical_adunit[i];
         }
   }

   // update cumulative cases per country
   for (i = 0; i < MAX_COUNTRIES; i++)
      State.cumC_country[i] = cumC_country[i];
   // update overall state variable for cumulative cases per adunit

   TimeSeries[n].rmsRad  = (State.cumI > 0) ? sqrt(State.sumRad2 / ((double)State.cumI)) : 0;
   TimeSeries[n].maxRad  = sqrt(State.maxRad2);
   TimeSeries[n].extinct = ((((g_allParams.SmallEpidemicCases >= 0) && (State.R <= g_allParams.SmallEpidemicCases))
                             || (g_allParams.SmallEpidemicCases < 0))
                            && (State.I + State.L == 0))
                              ? 1
                              : 0;
   for (i = 0; i < NUM_AGE_GROUPS; i++)
   {
      TimeSeries[n].incCa[i] = TimeSeries[n].incIa[i] = TimeSeries[n].incDa[i] = 0;
      for (j = 0; j < g_allParams.NumThreads; j++)
      {
         TimeSeries[n].incCa[i] += (double)StateT[j].cumCa[i];
         TimeSeries[n].incIa[i] += (double)StateT[j].cumIa[i];
         TimeSeries[n].incDa[i] += (double)StateT[j].cumDa[i];
      }
   }

   for (i = 0; i < 2; i++)
   {
      TimeSeries[n].incC_keyworker[i] = TimeSeries[n].incI_keyworker[i] = TimeSeries[n].cumT_keyworker[i] = 0;
      for (j = 0; j < g_allParams.NumThreads; j++)
      {
         TimeSeries[n].incC_keyworker[i] += (double)StateT[j].cumC_keyworker[i];
         TimeSeries[n].incI_keyworker[i] += (double)StateT[j].cumI_keyworker[i];
         TimeSeries[n].cumT_keyworker[i] += (double)StateT[j].cumT_keyworker[i];
         StateT[j].cumC_keyworker[i] = StateT[j].cumI_keyworker[i] = 0;
      }
   }

   for (i = 0; i < INFECT_TYPE_MASK; i++)
   {
      TimeSeries[n].incItype[i] = 0;
      for (j = 0; j < g_allParams.NumThreads; j++)
      {
         TimeSeries[n].incItype[i] += (double)StateT[j].cumItype[i];
         StateT[j].cumItype[i] = 0;
      }
   }
   if (g_allParams.DoAdUnits)
      for (i = 0; i <= g_allParams.NumAdunits; i++)
      {
         TimeSeries[n].incI_adunit[i] = TimeSeries[n].incC_adunit[i] = TimeSeries[n].cumT_adunit[i] =
            TimeSeries[n].incH_adunit[i] = TimeSeries[n].incDC_adunit[i] = TimeSeries[n].incCT_adunit[i] =
               TimeSeries[n].incDCT_adunit[i] = 0; // added detected cases: ggilani 03/02/15
         for (j = 0; j < g_allParams.NumThreads; j++)
         {
            TimeSeries[n].incI_adunit[i] += (double)StateT[j].cumI_adunit[i];
            TimeSeries[n].incC_adunit[i] += (double)StateT[j].cumC_adunit[i];
            TimeSeries[n].incDC_adunit[i] += (double)StateT[j].cumDC_adunit[i]; // added detected cases: ggilani
                                                                                // 03/02/15
            TimeSeries[n].incH_adunit[i] += (double)StateT[j].cumH_adunit[i]; // added hospitalisation
            TimeSeries[n].incCT_adunit[i] +=
               (double)StateT[j].cumCT_adunit[i]; // added contact tracing: ggilani 15/06/17
            TimeSeries[n].incCC_adunit[i] +=
               (double)StateT[j].cumCC_adunit[i]; // added cases who are contacts: ggilani 28/05/2019
            TimeSeries[n].incDCT_adunit[i] +=
               (double)StateT[j].cumDCT_adunit[i]; // added cases who are digitally contact traced: ggilani 11/03/20
            TimeSeries[n].cumT_adunit[i] += (double)StateT[j].cumT_adunit[i];
            State.cumC_adunit[i] += StateT[j].cumC_adunit[i];
            State.cumDC_adunit[i] += StateT[j].cumDC_adunit[i];
            StateT[j].cumI_adunit[i] = StateT[j].cumC_adunit[i] = StateT[j].cumH_adunit[i] = StateT[j].cumDC_adunit[i] =
               StateT[j].cumCT_adunit[i] = StateT[j].cumCC_adunit[i] = StateT[j].cumDCT_adunit[i] =
                  0; // added hospitalisation, detected cases, contact tracing: ggilani 03/02/15, 15/06/17
         }
         if (g_allParams.DoICUTriggers)
         {
            State.trigDC_adunit[i] += (int)TimeSeries[n].incCritical_adunit[i];
            if (n >= g_allParams.TriggersSamplingInterval)
               State.trigDC_adunit[i] -=
                  (int)TimeSeries[n - g_allParams.TriggersSamplingInterval].incCritical_adunit[i];
         }
         else
         {
            State.trigDC_adunit[i] += (int)TimeSeries[n].incDC_adunit[i];
            if (n >= g_allParams.TriggersSamplingInterval)
               State.trigDC_adunit[i] -= (int)TimeSeries[n - g_allParams.TriggersSamplingInterval].incDC_adunit[i];
         }
      }
   if (g_allParams.DoDigitalContactTracing)
      for (i = 0; i < g_allParams.NumAdunits; i++)
         TimeSeries[n].DCT_adunit[i] = (double)AdUnits[i].ndct; // added total numbers of contacts currently isolated
                                                                // due to digital contact tracing: ggilani 11/03/20
   if (g_allParams.DoPlaces)
      for (i = 0; i < NUM_PLACE_TYPES; i++)
      {
         numPC = 0;
         for (j = 0; j < g_allParams.Nplace[i]; j++)
            if (PLACE_CLOSED(i, j))
               numPC++;
         State.NumPlacesClosed[i]          = numPC;
         TimeSeries[n].PropPlacesClosed[i] = ((double)numPC) / ((double)g_allParams.Nplace[i]);
      }
   for (i = k = 0; i < g_allParams.microcellCount; i++)
      if (Mcells[i].socdist == 2)
         k++;
   TimeSeries[n].PropSocDist = ((double)k) / ((double)g_allParams.microcellCount);

   // update contact number distribution in State
   for (i = 0; i < (MAX_CONTACTS + 1); i++)
   {
      for (j = 0; j < g_allParams.NumThreads; j++)
      {
         State.contact_dist[i] += StateT[j].contact_dist[i];
         StateT[j].contact_dist[i] = 0;
      }
   }

   trigAlertC = State.cumDC;
   if (n >= g_allParams.PreControlClusterIdDuration)
      trigAlertC -= (int)TimeSeries[n - g_allParams.PreControlClusterIdDuration].cumDC;

   if (g_allParams.PreControlClusterIdUseDeaths)
   {
      trigAlert = (int)TimeSeries[n].D;
      if (n >= g_allParams.PreControlClusterIdDuration)
         trigAlert -= (int)TimeSeries[n - g_allParams.PreControlClusterIdDuration].D;
   }
   else
   {
      trigAlert = trigAlertC;
   }

   if (((!g_allParams.DoAlertTriggerAfterInterv) && (trigAlert >= g_allParams.PreControlClusterIdCaseThreshold))
       || ((g_allParams.DoAlertTriggerAfterInterv)
           && (((trigAlertC >= g_allParams.PreControlClusterIdCaseThreshold) && (g_allParams.ModelCalibIteration <= 4))
               || ((t >= g_allParams.PreIntervTime) && (g_allParams.ModelCalibIteration > 4)))))
   {
      if ((!g_allParams.StopCalibration) && (!InterruptRun))
      {
         if (g_allParams.PreControlClusterIdTime == 0)
         {
            g_allParams.PreIntervTime = g_allParams.PreControlClusterIdTime = t;
            if (g_allParams.PreControlClusterIdCalTime >= 0)
            {
               g_allParams.PreControlClusterIdHolOffset =
                  g_allParams.PreControlClusterIdTime - g_allParams.PreIntervIdCalTime;
               //					fprintf(stderr, "@@## trigAlertC=%i g_allParams.PreControlClusterIdHolOffset=%lg \n",trigAlertC,
               //g_allParams.PreControlClusterIdHolOffset);
            }
         }
         if ((g_allParams.PreControlClusterIdCalTime >= 0) && (!g_allParams.DoAlertTriggerAfterInterv))
         {
            g_allParams.StopCalibration = 1;
            InterruptRun                = 1;
         }
         if ((g_allParams.DoAlertTriggerAfterInterv)
             && (t
                 == g_allParams.PreControlClusterIdTime + g_allParams.PreControlClusterIdCalTime
                       - g_allParams.PreIntervIdCalTime))
         {
            if ((trigAlert > 0) && (g_allParams.ModelCalibIteration < 15))
            {
               s   = ((double)trigAlert) / ((double)g_allParams.AlertTriggerAfterIntervThreshold);
               thr = 1.1 / sqrt((double)g_allParams.AlertTriggerAfterIntervThreshold);
               if (thr < 0.05)
                  thr = 0.05;
               fprintf(stderr, "\n** %i %lf %lf | %lg / %lg \t", g_allParams.ModelCalibIteration, t,
                       g_allParams.PreControlClusterIdTime + g_allParams.PreControlClusterIdCalTime
                          - g_allParams.PreIntervIdCalTime,
                       g_allParams.PreControlClusterIdHolOffset, s);
               fprintf(stderr, "| %i %i %i %i -> ", trigAlert, trigAlertC, g_allParams.AlertTriggerAfterIntervThreshold,
                       g_allParams.PreControlClusterIdCaseThreshold);
               if (g_allParams.ModelCalibIteration == 1)
               {
                  if ((((s - 1.0) <= thr) && (s >= 1)) || (((1.0 - s) <= thr / 2) && (s < 1)))
                  {
                     g_allParams.ModelCalibIteration = 15;
                     g_allParams.StopCalibration     = 1;
                  }
                  else
                  {
                     s = pow(s, 1.0);
                     k = (int)(((double)g_allParams.PreControlClusterIdCaseThreshold) / s);
                     if (k > 0)
                        g_allParams.PreControlClusterIdCaseThreshold = k;
                  }
               }
               else if ((g_allParams.ModelCalibIteration >= 4) && ((g_allParams.ModelCalibIteration) % 2 == 0))
               {
                  if ((((s - 1.0) <= thr) && (s >= 1)) || (((1.0 - s) <= thr / 2) && (s < 1)))
                  {
                     // g_allParams.ModelCalibIteration=15;
                     // g_allParams.StopCalibration = 1;
                  }
                  else if (s > 1)
                  {
                     g_allParams.PreIntervTime--;
                     g_allParams.PreControlClusterIdHolOffset--;
                  }
                  else if (s < 1)
                  {
                     g_allParams.PreIntervTime++;
                     g_allParams.PreControlClusterIdHolOffset++;
                  }
               }
               else if ((g_allParams.ModelCalibIteration >= 4) && ((g_allParams.ModelCalibIteration) % 2 == 1))
               {
                  if ((((s - 1.0) <= thr) && (s >= 1)) || (((1.0 - s) <= thr / 2) && (s < 1)))
                  {
                     g_allParams.ModelCalibIteration = 15;
                     g_allParams.StopCalibration     = 1;
                     fprintf(stderr, "Calibration ended.\n");
                  }
                  else
                     g_allParams.SeedingScaling /= pow(s, 0.4);
               }
               g_allParams.ModelCalibIteration++;
               InterruptRun = 1;
               fprintf(stderr, "%i : %lg\n", g_allParams.PreControlClusterIdCaseThreshold, g_allParams.SeedingScaling);
            }
            else
            {
               g_allParams.StopCalibration = 1;
               InterruptRun                = 1;
            }
         }
      }
      g_allParams.ControlPropCasesId = g_allParams.PostAlertControlPropCasesId;

      if (g_allParams.VaryEfficaciesOverTime)
         UpdateEfficaciesAndComplianceProportions(t - g_allParams.PreIntervTime);

      //// Set Case isolation start time (by admin unit)
      for (i = 0; i < g_allParams.NumAdunits; i++)
         if (ChooseTriggerVariableAndValue(i) > ChooseThreshold(
                i, g_allParams
                      .CaseIsolation_CellIncThresh)) //// a little wasteful if doing Global trigs as function called
                                                     ///more times than necessary, but worth it for much simpler code.
                                                     ///Also this function is small portion of runtime.
         {
            if (g_allParams.DoInterventionDelaysByAdUnit)
               DoOrDontAmendStartTime(&AdUnits[i].CaseIsolationTimeStart, t + AdUnits[i].CaseIsolationDelay);
            else
               DoOrDontAmendStartTime(&AdUnits[i].CaseIsolationTimeStart, t + g_allParams.CaseIsolationTimeStartBase);
         }

      //// Set Household Quarantine start time (by admin unit)
      for (i = 0; i < g_allParams.NumAdunits; i++)
         if (ChooseTriggerVariableAndValue(i) > ChooseThreshold(
                i, g_allParams.HHQuar_CellIncThresh)) //// a little wasteful if doing Global trigs as function called
                                                      ///more times than necessary, but worth it for much simpler code.
                                                      ///Also this function is small portion of runtime.
         {
            if (g_allParams.DoInterventionDelaysByAdUnit)
               DoOrDontAmendStartTime(&AdUnits[i].HQuarantineTimeStart, t + AdUnits[i].HQuarantineDelay);
            else
               DoOrDontAmendStartTime(&AdUnits[i].HQuarantineTimeStart, t + g_allParams.HQuarantineTimeStartBase);
         }

      //// Set DigitalContactTracingTimeStart
      if (g_allParams.DoDigitalContactTracing)
         for (i = 0; i < g_allParams.NumAdunits; i++)
            if (ChooseTriggerVariableAndValue(i) > ChooseThreshold(
                   i, g_allParams.DigitalContactTracing_CellIncThresh)) //// a little wasteful if doing Global trigs as
                                                                        ///function called more times than necessary,
                                                                        ///but worth it for much simpler code. Also this
                                                                        ///function is small portion of runtime.
            {
               if (g_allParams.DoInterventionDelaysByAdUnit)
                  DoOrDontAmendStartTime(&AdUnits[i].DigitalContactTracingTimeStart, t + AdUnits[i].DCTDelay);
               else
                  DoOrDontAmendStartTime(&AdUnits[i].DigitalContactTracingTimeStart,
                                         t + g_allParams.DigitalContactTracingTimeStartBase);
            }

      if (g_allParams.DoGlobalTriggers)
      {
         int TriggerValue = ChooseTriggerVariableAndValue(0);
         if (TriggerValue >= ChooseThreshold(0, g_allParams.TreatCellIncThresh))
            DoOrDontAmendStartTime(&(g_allParams.TreatTimeStart), t + g_allParams.TreatTimeStartBase);
         if (TriggerValue >= g_allParams.VaccCellIncThresh)
            DoOrDontAmendStartTime(&g_allParams.VaccTimeStart, t + g_allParams.VaccTimeStartBase);
         if (TriggerValue >= g_allParams.SocDistCellIncThresh)
         {
            DoOrDontAmendStartTime(&g_allParams.SocDistTimeStart, t + g_allParams.SocDistTimeStartBase);
            // added this for admin unit based intervention delays based on a global trigger: ggilani 17/03/20
            if (g_allParams.DoInterventionDelaysByAdUnit)
               for (i = 0; i < g_allParams.NumAdunits; i++)
                  DoOrDontAmendStartTime(&AdUnits[i].SocialDistanceTimeStart, t + AdUnits[i].SocialDistanceDelay);
         }
         if (TriggerValue >= g_allParams.PlaceCloseCellIncThresh)
         {
            DoOrDontAmendStartTime(&g_allParams.PlaceCloseTimeStart, t + g_allParams.PlaceCloseTimeStartBase);
            if (g_allParams.DoInterventionDelaysByAdUnit)
               for (i = 0; i < g_allParams.NumAdunits; i++)
                  DoOrDontAmendStartTime(&AdUnits[i].PlaceCloseTimeStart, t + AdUnits[i].PlaceCloseDelay);
         }
         if (TriggerValue >= g_allParams.MoveRestrCellIncThresh)
            DoOrDontAmendStartTime(&g_allParams.MoveRestrTimeStart, t + g_allParams.MoveRestrTimeStartBase);
         if (TriggerValue >= g_allParams.KeyWorkerProphCellIncThresh)
            DoOrDontAmendStartTime(&g_allParams.KeyWorkerProphTimeStart, t + g_allParams.KeyWorkerProphTimeStartBase);
      }
      else
      {
         DoOrDontAmendStartTime(&g_allParams.TreatTimeStart, t + g_allParams.TreatTimeStartBase);
         DoOrDontAmendStartTime(&g_allParams.VaccTimeStart, t + g_allParams.VaccTimeStartBase);
         DoOrDontAmendStartTime(&g_allParams.SocDistTimeStart, t + g_allParams.SocDistTimeStartBase);
         DoOrDontAmendStartTime(&g_allParams.PlaceCloseTimeStart, t + g_allParams.PlaceCloseTimeStartBase);
         DoOrDontAmendStartTime(&g_allParams.MoveRestrTimeStart, t + g_allParams.MoveRestrTimeStartBase);
         DoOrDontAmendStartTime(&g_allParams.KeyWorkerProphTimeStart, t + g_allParams.KeyWorkerProphTimeStartBase);
      }
      DoOrDontAmendStartTime(&g_allParams.AirportCloseTimeStart, t + g_allParams.AirportCloseTimeStartBase);
   }
   if ((g_allParams.PlaceCloseIndepThresh > 0) && (((double)State.cumDC) >= g_allParams.PlaceCloseIndepThresh))
      DoOrDontAmendStartTime(&g_allParams.PlaceCloseTimeStart, t + g_allParams.PlaceCloseTimeStartBase);

   if (t > g_allParams.SocDistTimeStart + g_allParams.SocDistChangeDelay)
   {
      g_allParams.SocDistDurationCurrent                = g_allParams.SocDistDuration2;
      g_allParams.SocDistHouseholdEffectCurrent         = g_allParams.SocDistHouseholdEffect2;
      g_allParams.SocDistSpatialEffectCurrent           = g_allParams.SocDistSpatialEffect2;
      g_allParams.EnhancedSocDistHouseholdEffectCurrent = g_allParams.EnhancedSocDistHouseholdEffect2;
      g_allParams.EnhancedSocDistSpatialEffectCurrent   = g_allParams.EnhancedSocDistSpatialEffect2;
      for (i = 0; i < g_allParams.PlaceTypeNum; i++)
      {
         g_allParams.SocDistPlaceEffectCurrent[i]         = g_allParams.SocDistPlaceEffect2[i];
         g_allParams.EnhancedSocDistPlaceEffectCurrent[i] = g_allParams.EnhancedSocDistPlaceEffect2[i];
      }
   }
   // fix to switch off first place closure after g_allParams.PlaceCloseDuration has elapsed, if there are no school or
   // cell-based triggers set
   if (t == g_allParams.PlaceCloseTimeStart + g_allParams.PlaceCloseDuration)
   {
      g_allParams.PlaceCloseTimeStartPrevious = g_allParams.PlaceCloseTimeStart;
      if ((g_allParams.PlaceCloseIncTrig == 0) && (g_allParams.PlaceCloseFracIncTrig == 0)
          && (g_allParams.PlaceCloseCellIncThresh == 0))
         g_allParams.PlaceCloseTimeStart = 9e9;
   }

   if (!g_allParams.VaryEfficaciesOverTime)
   {
      if ((g_allParams.PlaceCloseTimeStart2 > g_allParams.PlaceCloseTimeStartPrevious)
          && //// if second place closure start time after previous start time AND
          (t >= g_allParams.PlaceCloseTimeStartPrevious + g_allParams.PlaceCloseDuration)
          && //// if now after previous place closure period has finished AND
          (t >= g_allParams.PlaceCloseTimeStartPrevious + g_allParams.PlaceCloseTimeStartBase2
                   - g_allParams.PlaceCloseTimeStartBase)) //// if now after previous start time + plus difference
                                                           ///between 1st and 2nd base start times
      {
         fprintf(stderr, "\nSecond place closure period (t=%lg)\n", t);
         g_allParams.PlaceCloseTimeStartPrevious = g_allParams.PlaceCloseTimeStart2 = g_allParams.PlaceCloseTimeStart =
            t;
         g_allParams.PlaceCloseDuration      = g_allParams.PlaceCloseDuration2;
         g_allParams.PlaceCloseIncTrig       = g_allParams.PlaceCloseIncTrig2;
         g_allParams.PlaceCloseCellIncThresh = g_allParams.PlaceCloseCellIncThresh2;
      }
   }

   if (g_allParams.OutputBitmap >= 1)
   {
      TSMean = TSMeanNE;
      TSVar  = TSVarNE;
      CaptureBitmap();
      OutputBitmap(0);
   }
}

void RecordInfTypes(void)
{
   // int i, j, k, l, lc, lc2, b, c, n, nf, i2;

   // TODO: Look into this guy.  It gets aassigned various values that are never used.
   int lc = 0;

   double *res = nullptr;
   double *res_av = nullptr;
   double *res_var = nullptr;
   double t = 0.0;
   double s = 0.0;

   for (int n = 0; n < g_allParams.totalSampleNumber; n++)
   {
      for (int i = 0; i < INFECT_TYPE_MASK; i++)
         TimeSeries[n].Rtype[i] = 0;

      for (int i = 0; i < NUM_AGE_GROUPS; i++)
         TimeSeries[n].Rage[i] = 0;

      TimeSeries[n].Rdenom = 0;
   }

   for (int i = 0; i < INFECT_TYPE_MASK; i++)
      inftype[i] = 0;

   for (int i = 0; i < MAX_COUNTRIES; i++)
      infcountry[i] = 0;

   for (int i = 0; i < MAX_SEC_REC; i++)
      for (int j = 0; j < MAX_GEN_REC; j++)
         indivR0[i][j] = 0;

   for (int i = 0; i <= MAX_HOUSEHOLD_SIZE; i++)
      for (int j = 0; j <= MAX_HOUSEHOLD_SIZE; j++)
         inf_household[i][j] = case_household[i][j] = 0;

   for (int b = 0; b < g_allParams.cellCount; b++)
      if ((Cells[b].S != Cells[b].n) || (Cells[b].R > 0))
         for (int c = 0; c < Cells[b].n; c++)
            Hosts[Cells[b].members[c]].listpos = 0;

   //	for(b=0;b<g_allParams.NC;b++)
   //		if((Cells[b].S!=Cells[b].n)||(Cells[b].R>0))
   {
      int j = 0;
      int k = 0;
      int l = 0;
      int lc = 0;
      int lc2 = 0;
      t = 1e10;

      //			for(c=0;c<Cells[b].n;c++)
      for (int i = 0; i < g_allParams.populationSize; i++)
      {
         //				i=Cells[b].members[c];
         if (j == 0)
         {
            j = Households[Hosts[i].hh].nh;
            k = j;
         }

         if ((Hosts[i].inf != InfStat_Susceptible) && (Hosts[i].inf != InfStat_ImmuneAtStart))
         {
            if (Hosts[i].latent_time * g_allParams.TimeStep <= g_allParams.SampleTime)
            {
               TimeSeries[(int)(Hosts[i].latent_time * g_allParams.TimeStep / g_allParams.SampleStep)].Rdenom++;
            }

            infcountry[Mcells[Hosts[i].mcell].country]++;

            if (abs(Hosts[i].inf) < InfStat_Recovered)
            {
               l = -1;
            }
            else if (l >= 0)
            {
               l++;
            }

            if ((l >= 0) && ((Hosts[i].inf == InfStat_RecoveredFromSymp) || (Hosts[i].inf == InfStat_Dead_WasSymp)))
            {
               lc2++;
               if (Hosts[i].latent_time * g_allParams.TimeStep
                   <= t) // This convoluted logic is to pick up households where the index is symptomatic
               {
                  lc = 1;
                  t  = Hosts[i].latent_time * g_allParams.TimeStep;
               }
            }
            else if ((l > 0) && (Hosts[i].latent_time * g_allParams.TimeStep < t))
            {
               lc = 0;
               t  = Hosts[i].latent_time * g_allParams.TimeStep;
            }

            int i2 = Hosts[i].infector;
            if (i2 >= 0)
            {
               Hosts[i2].listpos++;
               if (Hosts[i2].latent_time * g_allParams.TimeStep <= g_allParams.SampleTime)
               {
                  TimeSeries[(int)(Hosts[i2].latent_time * g_allParams.TimeStep / g_allParams.SampleStep)]
                     .Rtype[Hosts[i].infect_type % INFECT_TYPE_MASK]++;
                  TimeSeries[(int)(Hosts[i2].latent_time * g_allParams.TimeStep / g_allParams.SampleStep)]
                     .Rage[HOST_AGE_GROUP(i)]++;
               }
            }
         }

         inftype[Hosts[i].infect_type % INFECT_TYPE_MASK]++;
         j--;
         if (j == 0)
         {
            if (l < 0)
            {
               l = 0;
            }

            inf_household[k][l]++;
            case_household[k][lc2]++; // now recording total symptomatic cases, rather than infections conditional on
                                      // symptomatic index
            l   = 0;
            lc  = 0;
            lc2 = 0;
            t   = 1e10;
         }
      }
   }
   for (int b = 0; b < g_allParams.cellCount; b++)
   {
      if ((Cells[b].S != Cells[b].n) || (Cells[b].R > 0))
      {
         for (int c = 0; c < Cells[b].n; c++)
         {
            int i = Cells[b].members[c];
            if ((abs(Hosts[i].inf) == InfStat_Recovered) || (abs(Hosts[i].inf) == InfStat_Dead))
            {
               int l = Hosts[i].infect_type / INFECT_TYPE_MASK;
               if ((l < MAX_GEN_REC) && (Hosts[i].listpos < MAX_SEC_REC))
               {
                  indivR0[Hosts[i].listpos][l]++;
               }
            }
         }
      }
   }
   /* 	if(!TimeSeries[g_allParams.NumSamples-1].extinct) */

   {
      for (int i = 0; i < INFECT_TYPE_MASK; i++)
         inftype_av[i] += inftype[i];

      for (int i = 0; i < MAX_COUNTRIES; i++)
      {
         infcountry_av[i] += infcountry[i];
         if (infcountry[i] > 0)
         {
            infcountry_num[i]++;
         }
      }

      for (int i = 0; i < MAX_SEC_REC; i++)
      {
         for (int j = 0; j < MAX_GEN_REC; j++)
         {
            indivR0_av[i][j] += indivR0[i][j];
         }
      }

      for (int i = 0; i <= MAX_HOUSEHOLD_SIZE; i++)
      {
         for (int j = 0; j <= MAX_HOUSEHOLD_SIZE; j++)
         {
            inf_household_av[i][j] += inf_household[i][j];
            case_household_av[i][j] += case_household[i][j];
         }
      }
   }

   int k = (int)(g_allParams.PreIntervIdCalTime - g_allParams.PreControlClusterIdTime);
   for (int n = 0; n < g_allParams.totalSampleNumber; n++)
   {
      TimeSeries[n].t += k;
      if (TimeSeries[n].Rdenom == 0)
         TimeSeries[n].Rdenom = 1e-10;
      for (int i = 0; i < NUM_AGE_GROUPS; i++)
         TimeSeries[n].Rage[i] /= TimeSeries[n].Rdenom;

      s = 0;
      for (int i = 0; i < INFECT_TYPE_MASK; i++)
         s += (TimeSeries[n].Rtype[i] /= TimeSeries[n].Rdenom);

      TimeSeries[n].Rdenom = s;
   }

   int nf = sizeof(results) / sizeof(double);
   if (!g_allParams.DoAdUnits)
      nf -= MAX_ADUNITS; // TODO: This still processes most of the AdUnit arrays; just not the last one

   fprintf(stderr, "extinct=%i (%i)\n", (int)TimeSeries[g_allParams.totalSampleNumber - 1].extinct,
           g_allParams.totalSampleNumber - 1);
   if (TimeSeries[g_allParams.totalSampleNumber - 1].extinct)
   {
      TSMean = TSMeanE;
      TSVar  = TSVarE;
      g_allParams.NRactE++;
   }
   else
   {
      TSMean = TSMeanNE;
      TSVar  = TSVarNE;
      g_allParams.NRactNE++;
   }

   lc = -k;
   for (int n = 0; n < g_allParams.totalSampleNumber; n++)
   {
      if ((n + lc >= 0) && (n + lc < g_allParams.totalSampleNumber))
      {
         if (s < TimeSeries[n + lc].incC)
         {
            s = TimeSeries[n + lc].incC;
            t = g_allParams.SampleStep * ((double)(n + lc));
         }
         res     = (double*)&TimeSeries[n + lc];
         res_av  = (double*)&TSMean[n];
         res_var = (double*)&TSVar[n];
         for (int i = 1 /* skip over `t` */; i < nf; i++)
         {
            res_av[i] += res[i];
            res_var[i] += res[i] * res[i];
         }
         if (TSMean[n].cumTmax < TimeSeries[n + lc].cumT)
            TSMean[n].cumTmax = TimeSeries[n + lc].cumT;
         if (TSMean[n].cumVmax < TimeSeries[n + lc].cumV)
            TSMean[n].cumVmax = TimeSeries[n + lc].cumV;
      }
      TSMean[n].t += ((double)n) * g_allParams.SampleStep;
   }

   PeakHeightSum += s;
   PeakHeightSS += s * s;
   PeakTimeSum += t;
   PeakTimeSS += t * t;
}

void CalcOriginDestMatrix_adunit()
{
   /** function: CalcOriginDestMatrix_adunit()
    *
    * purpose: to output the origin destination matrix between admin units
    *
    * parameters: none
    *
    * returns: none
    *
    * author: ggilani, date: 28/01/15
    */
   int tn, i, j, k, l, m, p;
   double total_flow, flow;
   ptrdiff_t cl_from, cl_to, cl_from_mcl, cl_to_mcl, mcl_from, mcl_to;

#pragma omp parallel for private(tn, i, j, k, l, m, p, total_flow, mcl_from, mcl_to, cl_from, cl_to, cl_from_mcl, \
                                 cl_to_mcl, flow) schedule(static) // reduction(+:s,t2)
   for (tn = 0; tn < g_allParams.NumThreads; tn++)
   {
      for (i = tn; i < g_allParams.populatedCellCount; i += g_allParams.NumThreads)
      {
         // reset pop density matrix to zero
         double pop_dens_from[MAX_ADUNITS] = {};

         // find index of cell from which flow travels
         cl_from     = CellLookup[i] - Cells;
         cl_from_mcl = (cl_from / g_allParams.nch) * g_allParams.microcellsOnACellSide * g_allParams.nmch
                       + (cl_from % g_allParams.nch) * g_allParams.microcellsOnACellSide;

         // loop over microcells in these cells to find populations in each admin unit and so flows
         for (k = 0; k < g_allParams.microcellsOnACellSide; k++)
         {
            for (l = 0; l < g_allParams.microcellsOnACellSide; l++)
            {
               // get index of microcell
               mcl_from = cl_from_mcl + l + k * g_allParams.nmch;
               if (Mcells[mcl_from].n > 0)
               {
                  // get proportion of each population of cell that exists in each admin unit
                  pop_dens_from[Mcells[mcl_from].adunit] += (((double)Mcells[mcl_from].n) / ((double)Cells[cl_from].n));
               }
            }
         }

         for (j = i; j < g_allParams.populatedCellCount; j++)
         {
            // reset pop density matrix to zero
            double pop_dens_to[MAX_ADUNITS] = {};

            // find index of cell which flow travels to
            cl_to     = CellLookup[j] - Cells;
            cl_to_mcl = (cl_to / g_allParams.nch) * g_allParams.microcellsOnACellSide * g_allParams.nmch
                        + (cl_to % g_allParams.nch) * g_allParams.microcellsOnACellSide;
            // calculate distance and kernel between the cells
            // total_flow=Cells[cl_from].max_trans[j]*Cells[cl_from].n*Cells[cl_to].n;
            if (j == 0)
            {
               total_flow = Cells[cl_from].cum_trans[j] * Cells[cl_from].n;
            }
            else
            {
               total_flow = (Cells[cl_from].cum_trans[j] - Cells[cl_from].cum_trans[j - 1]) * Cells[cl_from].n;
            }

            // loop over microcells within destination cell
            for (m = 0; m < g_allParams.microcellsOnACellSide; m++)
            {
               for (p = 0; p < g_allParams.microcellsOnACellSide; p++)
               {
                  // get index of microcell
                  mcl_to = cl_to_mcl + p + m * g_allParams.nmch;
                  if (Mcells[mcl_to].n > 0)
                  {
                     // get proportion of each population of cell that exists in each admin unit
                     pop_dens_to[Mcells[mcl_to].adunit] += (((double)Mcells[mcl_to].n) / ((double)Cells[cl_to].n));
                  }
               }
            }

            for (m = 0; m < g_allParams.NumAdunits; m++)
            {
               for (p = 0; p < g_allParams.NumAdunits; p++)
               {
                  if (m != p)
                  {
                     flow = total_flow * pop_dens_from[m]
                            * pop_dens_to[p]; // updated to remove reference to cross-border flows: ggilani 26/03/20
                     StateT[tn].origin_dest[m][p] += flow;
                     StateT[tn].origin_dest[p][m] += flow;
                  }
               }
            }
         }

         ////loop over microcells within cell to find the proportion of the cell population in each admin unit
         // k=(cl_from/g_allParams.nch)*g_allParams.NMCL*g_allParams.nmch+(cl_from%g_allParams.nch)*g_allParams.NMCL;
         // for(l=0;l<g_allParams.NMCL;l++)
         //{
         //	for(m=0;m<g_allParams.NMCL;m++)
         //	{
         //		mcl_from=k+m+l*g_allParams.nmch;
         //		pop_cell_from[Mcells[mcl_from].adunit]+=Mcells[mcl_from].n;
         //	}
         //}
         ////loop over cells
         // for(p=(i+1);p<g_allParams.NCP;p++)
         //{
         //	//reset population array
         //	for(j=0;j<g_allParams.NumAdunits;j++)
         //	{
         //		pop_cell_to[j]=0.0;
         //	}
         //	cl_to=CellLookup[p]-Cells;
         //	//loop over microcells within cell to find the proportion of the cell population in each admin unit
         //	q=(cl_to/g_allParams.nch)*g_allParams.NMCL*g_allParams.nmch+(cl_to%g_allParams.nch)*g_allParams.NMCL;
         //	for(l=0;l<g_allParams.NMCL;l++)
         //	{
         //		for(m=0;m<g_allParams.NMCL;m++)
         //		{
         //			mcl_to=q+m+l*g_allParams.nmch;
         //			pop_cell_to[Mcells[mcl_to].adunit]+=Mcells[mcl_to].n;
         //		}
         //	}

         //	//find distance and kernel function between cells
         //	dist=dist2_cc_min(Cells+cl_from,Cells+cl_to);
         //	dist_kernel=numKernel(dist);

         //	//add flow between adunits based on how population is distributed
         //	for(l=0;l<g_allParams.NumAdunits;l++)
         //	{
         //		for(m=(l+1);m<g_allParams.NumAdunits;m++)
         //		{
         //			AdUnits[l].origin_dest[m]+=pop_cell_from[l]*pop_cell_to[m]*dist_kernel;
         //			AdUnits[m].origin_dest[l]+=pop_cell_from[l]*pop_cell_to[m]*dist_kernel;
         //		}
         //	}
      }
   }

   // Sum up flow between adunits across threads
   for (i = 0; i < g_allParams.NumAdunits; i++)
   {
      for (j = 0; j < g_allParams.NumAdunits; j++)
      {
         for (k = 0; k < g_allParams.NumThreads; k++)
         {
            AdUnits[i].origin_dest[j] += StateT[k].origin_dest[i][j];
         }
      }
   }
}

//// Get parameters code (called by ReadParams function)
int GetInputParameter(FILE* dat,
                      FILE* dat2,
                      const char* SItemName,
                      const char* ItemType,
                      void* ItemPtr,
                      int NumItem,
                      int NumItem2,
                      int Offset)
{
   int FindFlag;

   FindFlag = GetInputParameter2(dat, dat2, SItemName, ItemType, ItemPtr, NumItem, NumItem2, Offset);
   if (!FindFlag)
   {
      ERR_CRITICAL_FMT("\nUnable to find parameter `%s' in input file. Aborting program...\n", SItemName);
   }

   return FindFlag;
}

int GetInputParameter2(FILE* dat,
                       FILE* dat2,
                       const char* SItemName,
                       const char* ItemType,
                       void* ItemPtr,
                       int NumItem,
                       int NumItem2,
                       int Offset)
{
   int FindFlag = 0;

   if (dat2)
      FindFlag = GetInputParameter3(dat2, SItemName, ItemType, ItemPtr, NumItem, NumItem2, Offset);
   if (!FindFlag)
      FindFlag = GetInputParameter3(dat, SItemName, ItemType, ItemPtr, NumItem, NumItem2, Offset);
   return FindFlag;
}

/*
    Reads a string (as per fscanf %s).
    Returns true if it succeeds, false on EOF, and does not return on error.
*/
bool readString(const char* SItemName, FILE* dat, char* buf)
{
   int r = fscanf(dat, "%s", buf);
   if (r == 1)
   {
      return true;
   }
   else if (r == EOF)
   {
      if (ferror(dat))
      {
         ERR_CRITICAL_FMT("fscanf failed for %s: %s.\n", SItemName, strerror(errno));
      }
      else
      {
         // EOF
         return false;
      }
   }
   else
   {
      ERR_CRITICAL_FMT("Unexpected fscanf result %d for %s.\n", r, SItemName);
   }
}

int GetInputParameter3(FILE* dat,
                       const char* SItemName,
                       const char* ItemType,
                       void* ItemPtr,
                       int NumItem,
                       int NumItem2,
                       int Offset)
{
   char match[10000]        = "";
   char ReadItemName[10000] = "";
   char ItemName[10000]     = {};

   int FindFlag = 0;
   int EndString = -1;
   int CurPos = -1;

   int n = 0;
   fseek(dat, 0, 0);
   sprintf(ItemName, "[%s]", SItemName);
   while (!FindFlag)
   {
      if (!readString(SItemName, dat, match))
         return 0;

      FindFlag = (!strncmp(match, ItemName, strlen(match)));
      if (FindFlag)
      {
         CurPos = ftell(dat);
         strcpy(ReadItemName, match);
         EndString = (match[strlen(match) - 1] == ']');
         while ((!EndString) && (FindFlag))
         {
            if (!readString(SItemName, dat, match))
               return 0;
            strcat(ReadItemName, " ");
            strcat(ReadItemName, match);
            FindFlag  = (!strncmp(ReadItemName, ItemName, strlen(ReadItemName)));
            EndString = (ReadItemName[strlen(ReadItemName) - 1] == ']');
         }
         if (!EndString)
         {
            fseek(dat, CurPos, 0);
            FindFlag = 0;
         }
      }
   }
   if (FindFlag)
   {
      FindFlag = 0;
      if (!strcmp(ItemType, "%lf"))
         n = 1;
      else if (!strcmp(ItemType, "%i"))
         n = 2;
      else if (!strcmp(ItemType, "%s"))
         n = 3;
      if (NumItem2 < 2)
      {
         if (NumItem == 1)
         {
            if (fscanf(dat, "%s", match) != 1)
            {
               ERR_CRITICAL_FMT("fscanf failed for %s\n", SItemName);
            }
            if ((match[0] == '#') && (match[1] == '1'))
            {
               FindFlag++;
               if (n == 1)
                  *((double*)ItemPtr) = g_allParams.clP1;
               else if (n == 2)
                  *((int*)ItemPtr) = (int)g_allParams.clP1;
               else if (n == 3)
                  sscanf(match, "%s", (char*)ItemPtr);
            }
            else if ((match[0] == '#') && (match[1] == '2'))
            {
               FindFlag++;
               if (n == 1)
                  *((double*)ItemPtr) = g_allParams.clP2;
               else if (n == 2)
                  *((int*)ItemPtr) = (int)g_allParams.clP2;
               else if (n == 3)
                  sscanf(match, "%s", (char*)ItemPtr);
            }
            else if ((match[0] == '#') && (match[1] == '3'))
            {
               FindFlag++;
               if (n == 1)
                  *((double*)ItemPtr) = g_allParams.clP3;
               else if (n == 2)
                  *((int*)ItemPtr) = (int)g_allParams.clP3;
               else if (n == 3)
                  sscanf(match, "%s", (char*)ItemPtr);
            }
            else if ((match[0] == '#') && (match[1] == '4'))
            {
               FindFlag++;
               if (n == 1)
                  *((double*)ItemPtr) = g_allParams.clP4;
               else if (n == 2)
                  *((int*)ItemPtr) = (int)g_allParams.clP4;
               else if (n == 3)
                  sscanf(match, "%s", (char*)ItemPtr);
            }
            else if ((match[0] == '#') && (match[1] == '5'))
            {
               FindFlag++;
               if (n == 1)
                  *((double*)ItemPtr) = g_allParams.clP5;
               else if (n == 2)
                  *((int*)ItemPtr) = (int)g_allParams.clP5;
               else if (n == 3)
                  sscanf(match, "%s", (char*)ItemPtr);
            }
            else if ((match[0] == '#') && (match[1] == '6'))
            {
               FindFlag++;
               if (n == 1)
                  *((double*)ItemPtr) = g_allParams.clP6;
               else if (n == 2)
                  *((int*)ItemPtr) = (int)g_allParams.clP6;
               else if (n == 3)
                  sscanf(match, "%s", (char*)ItemPtr);
            }
            else if ((match[0] != '[') && (!feof(dat)))
            {
               FindFlag++;
               if (n == 1)
                  sscanf(match, "%lf", (double*)ItemPtr);
               else if (n == 2)
                  sscanf(match, "%i", (int*)ItemPtr);
               else if (n == 3)
                  sscanf(match, "%s", (char*)ItemPtr);
            }
         }
         else
         {
            for (CurPos = 0; CurPos < NumItem; CurPos++)
            {
               if (fscanf(dat, "%s", match) != 1)
               {
                  ERR_CRITICAL_FMT("fscanf failed for %s\n", SItemName);
               }
               if ((match[0] != '[') && (!feof(dat)))
               {
                  FindFlag++;
                  if (n == 1)
                     sscanf(match, "%lf", ((double*)ItemPtr) + CurPos + Offset);
                  else if (n == 2)
                     sscanf(match, "%i", ((int*)ItemPtr) + CurPos + Offset);
                  else if (n == 3)
                     sscanf(match, "%s", *(((char**)ItemPtr) + CurPos + Offset));
               }
               else
                  CurPos = NumItem;
            }
         }
      }
      else
      {
         for (int j = 0; j < NumItem; j++)
         { // added these braces
            for (int i = 0; i < NumItem2; i++)
            {
               if (fscanf(dat, "%s", match) != 1)
               {
                  ERR_CRITICAL_FMT("fscanf failed for %s\n", SItemName);
               }
               if ((match[0] != '[') && (!feof(dat)))
               {
                  FindFlag++;
                  if (n == 1)
                     sscanf(match, "%lf",
                            ((double**)ItemPtr)[j + Offset] + i
                               + Offset); // changed from [j+Offset]+i+Offset to +j+Offset+i, as ItemPtr isn't an array
                                          // - 01/10: changed it back
                  else
                     sscanf(match, "%i", ((int**)ItemPtr)[j + Offset] + i + Offset);
               }
               else
               {
                  i = NumItem2;
                  j = NumItem;
               }
            }
            // Offset=Offset+(NumItem2-1); //added this line to get the correct offset in address position when
            // incrementing j
         } // added these braces
      }
   }
   //	fprintf(stderr,"%s\n",SItemName);
   return FindFlag;
}
