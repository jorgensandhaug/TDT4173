
### t_1000hPa:K	
Maybe we can use this together with wind speed inverse or something, to model of efficiency of the solar panel, because higher wind will move air around the panels faster.

Same for direct_rad

### wind_speed_u_10m:ms	wind_speed_v_10m:ms	wind_speed_w_1000hPa:ms
can be dropped, since we have wind_speed_10m:ms

### azimuth:d elevation:d
These can maybe be transformed using sin and cos, and maybe combine them?

### snow and precipitation
I think we only should keep snow_depth:cm (for reflection) and fresh_snow_1h:cm

snow_density:kgm3 
has a ton of missing values. Maybe we should set those to zero. Maybe it is so that when the snow density is above a certain value it covers the solar panels. 

Snow_drift is always 0 for loc A, so we can drop that i think

lets try to drop all precip

### pressure
drop it?

### radiation
use only last hour, not effect (W)
create global_rad_1h:J, which is the sum of direct_rad_1h:J and diffuse_rad_1h:J

drop clear_sky_energy_1h:J	and clear_sky_rad:W




### notes
no variables should be negative, so we can set all negative values to 0

### cloud height
drop them


### dew_or_rime:idx
Change this to one variable for is_dew and one variable for is_rime

### is_in_shadow:idx and is_day:idx
drop them, we have better variables for this, e.g. effective_cloud_cover:p (dont use total_cloud_cover:p)





now, having these columns:
absolute_humidity_2m:gm3	air_density_2m:kgm3	ceiling_height_agl:m	clear_sky_energy_1h:J	clear_sky_rad:W	cloud_base_agl:m	dew_or_rime:idx	dew_point_2m:K	diffuse_rad:W	diffuse_rad_1h:J	direct_rad:W	direct_rad_1h:J	effective_cloud_cover:p	elevation:m	fresh_snow_12h:cm	fresh_snow_1h:cm	fresh_snow_24h:cm	fresh_snow_3h:cm	fresh_snow_6h:cm	is_day:idx	is_in_shadow:idx	msl_pressure:hPa	precip_5min:mm	precip_type_5min:idx	pressure_100m:hPa	pressure_50m:hPa	prob_rime:p	rain_water:kgm2	relative_humidity_1000hPa:p	sfc_pressure:hPa	snow_density:kgm3	snow_depth:cm	snow_drift:idx	snow_melt_10min:mm	snow_water:kgm2	sun_azimuth:d	sun_elevation:d	super_cooled_liquid_water:kgm2	t_1000hPa:K	total_cloud_cover:p	visibility:m	wind_speed_10m:ms	wind_speed_u_10m:ms	wind_speed_v_10m:ms	wind_speed_w_1000hPa:ms


Help me implement the things I've talked about above.