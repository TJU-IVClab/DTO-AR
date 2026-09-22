#pragma once




// Image measurement structure
typedef struct	POB_T		// Image observation
{
	long	phoid;			// Photo id.
	long	tarid;			// Target id.
	double	pob_pix[2];		// Image x,y coordinates in pixels
	double	pob[2];			// Image x,y coordinates in photocoordinate system and mm
	double	stdev[2];		// Image x,y precisions
	double	res[2];			// Image (RMS) x,y residuals
	double ellipse[3];	// Ellipse parameters (semi-major axis, semi-minor axis and orientation) for the target image when available
	// Otherwise just the extent in x and y with a zero orientation
	double	quality;		// Quality estimator for photo observation sorting - eg image correlation data
	long	epoch_id;		// Epoch number for the photo obs set
	long	temp_id;		// ID of linked element
	short	flag;			// Flag for image validity		 1=OK/valid
	//								 0=rejected
	//								-1=deleted
	//		Internal to photo.cpp	 2=Zheng-Wang resection point
	//								-2=invalid resection point
	short	type_flag;		// Flag for type of point, used to destinguish between data extracted from point measurement and edges

	int     idxInPool;
}	POB;



// Sub image structure
typedef struct	IMWIN_T		// Image window used for target location
{
	unsigned char** ucBuf;	// Image window pixel array - 8 bit  access by ucBuf[y][x]
	unsigned short** usBuf;	// 16 bit
	short	h_win;			// # columns in pixels
	short	v_win;			// # rows in pixels
	long	cur_x;			// Cursor click or approx location column
	long	cur_y;			// Cursor click or approx location row
}	IMWIN;


// Image location and measurement parameter structure
typedef struct	LOCATE_T
{
	char	locator;		// Options are 0, 1, 2 or m type
	char	threshold;		// Options are a, p, r or w type
	char	shape;			// Options are c, h, v
	long	constant;		// Grey level for additive constant threshold
	double	level;			// Sigma level for random threshold
	short	range;			// Critical value for max-min grey scale
	long	min_targ_size;	// Minimum span of target image
	short	ratio_test;		// Flag for xy ratio test
	double	ratio;			// Critical value of xy ratio test
	short	bw_test;		// Flag for bw ratio test
	short	perimeter_test;		// Flag for perimeter edge to target image span test
	double	perimeter_factor;	// Factor for the perimeter test (multiplied by minimum target perimeter)
	double	fMinBWR;		// Minimum black to white ratio (e.g. f (fMinBWR < 0.45 || fMaxBWR > 1.05) // Optimum limits
	double	fMaxBWR;		// Maximum black to white ratio	(e.g. (fMinBWR < 0.40 || fMaxBWR > 1.1) // Less restrictive limits
	//									(e.g. (fMinBWR < 0.67 || fMaxBWR > 0.91)	// Highly restrictive limits prevents code bar inclusion
}	LOCATE;



// Least squares estimation structure
typedef	struct	LSE_T
{
	double	ans_vec[6];		// Vector of increments to parameters of LSA
	double	des_mat[2][6];	// Design matrix rows for two equations per image
	double	nor_mat[6][6];	// Normals array matrix
	double	nor_vec[6];		// Normals array vector
	double	qq_meas[3];		// Weight coefficient matrix for image measurements
	double	meas[3];		// Measurement vector for one point measurement
	double	res[3];			// Residuals of observation equations
}	LSE;


short __declspec(dllexport)	get_location(POB* Coords, IMWIN* Image_win, LOCATE* Location, int* threshold, long* x_size, long* y_size);
