# -*- coding: utf-8 -*-
"""
routeur made by Etienne Clopeau

"""
#imports
from numpy import arange,cos,sin,radians, degrees, sqrt, arctan2, arctan
from numpy import array
import numpy as np

#matplotlib
#%matplotlib inline
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap
#import mpld3
#mpld3.enable_notebook()

#scipy
from scipy.interpolate import interp2d


from shapely.geometry import Polygon,Point
from shapely.ops import cascaded_union
from descartes import PolygonPatch

import pyproj

import time

# complet algo

# initial point
lat0, lon0 = -45 , -45.
latF, lonF = -20,20
timeStep = 1.
# azimut
az = 45.


class GRIB():
    def __init__(self):
        # load wind grib
        self.latitudes = arange(-90.,90.,10.)
        self.longitudes = arange(-180.,180.,10.)
        self.wind_directions = array([[b for b in self.longitudes] for a in self.latitudes])
        self.wind_velocities = array([[2*a*b/360.*sin(4*b)+5 for b in self.longitudes] for a in self.latitudes])
        self.windU = self.wind_velocities*cos(radians(self.wind_directions))
        self.windV = self.wind_velocities*sin(radians(self.wind_directions))

        self.gribInterpolatorU = interp2d(self.longitudes,self.latitudes,self.windU, kind = "linear")
        self.gribInterpolatorV = interp2d(self.longitudes,self.latitudes,self.windV, kind = "linear")

        
    def interpol(self,lon,lat):
        """lon in degrees
            lat in degrees
            return wind direction in degrees and wind velocity in noeuds"""
        u,v = self.gribInterpolatorU(lon, lat),self.gribInterpolatorV(lon, lat)
        return degrees(arctan2(v,u)), sqrt(u*u+v*v)
                
    def plot(self):
        #plots winds
        fig, ax = plt.subplots(figsize=(10, 10))
        
        lat, lon = np.meshgrid(self.latitudes, self.longitudes)
        ax.barbs(self.longitudes, self.latitudes,self.windU,self.windV, self.wind_velocities, pivot='middle')
        #ax.drawcoastlines()
        
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        
        fig.show()
       
        
class POLAR():
    def __init__(self, fileName):
        """load the polar in fileName"""
        f = open(fileName,'r')
        windAngle = list()
        speeds = list()
        for i,line in enumerate(f):
            if i==0:
                windSpeed = [float(a) for a in line.split(';')[1:]]
            else:
                vals = line.split(';')
                windAngle.append(float(vals[0]))
                speeds.append([float(a) for a in vals[1:]])
        self.windAngles = array(windAngle)
        self.windSpeed = array(windSpeed)
        self.speeds = array(speeds)
        
        self.interpolator = interp2d(self.windAngles,self.windSpeed,self.speeds.transpose(), kind = "linear")

    def plot(self):
        # plot of the polar
        fig = plt.figure(figsize=(12,5))
        ax1 = fig.add_subplot(121, projection = "polar")
        ax2 = fig.add_subplot(122)
        
        #ax1.plot(radians(polaire['windAngles']),polaire['speeds'][:,1])
        #print(polaire['speeds'][:,1])
        #print(polaire['windAngles'])
        for i,ws in enumerate(self.windSpeed):
            ax1.plot(radians(self.windAngles),self.speeds[:,i], label = "%s"%ws)
        ax1.set_xlabel("windAngle (°)")
        ax1.set_ylabel("boatSpeed")
        
        ax2.contourf(self.windAngles,self.windSpeed,self.speeds.transpose())
        ax2.set_xlabel("WindAngle [°]")
        ax2.set_ylabel("windSpeed")
            
        fig.show()


    def interpol(self, windAngle, windSpeed):
        """ 
        wind angle in degrees
        windSpeed en noeuds
        return boat speed en noeuds
        """
        windAngle = windAngle%360.
        if windAngle > 180. : windAngle = windAngle-360.
#        windAngle = windAngle-(windAngle>180.)*360.
        return self.interpolator(abs(windAngle),windSpeed)
#        return [self.interpolator(abs(wa),windSpeed) for wa in windAngle]

    
    

def mergePoly(polys,refpoly = None):
    polygons = [Polygon(p) for p in polys]
    u = cascaded_union(polygons)
    if refpoly is not None:
        u = cascaded_union([u,Polygon(refpoly)])
#    if len(polys) == 1: return polys[0]
#    
#    print('---mergePoly---')
#    print('nb poly',len(polys))
#
#    mergedPoly = list()
#    print("first poly")
#    for point in polys[0].exterior.coords:
#        point = Point(point)
#        if refpoly is not None and refpoly.contains(point): 
#            print('point contained in ref',point.coords[0]) 
#            continue
#        if polys[1].contains(point): 
#            print('point contained in next',point.coords[0]) 
#            continue
#        print('point added',point.coords[0]) 
#        mergedPoly.append(point.coords[0])
#    for poly_m1, poly, poly_p1 in zip(polys, polys[1:], polys[2:]):
#        print("next poly")
#        for point in poly.exterior.coords:
#            point = Point(point)
#            if poly_m1.contains(point):             
#                print('point contained in previous',point.coords[0]) 
#                continue
#            if refpoly is not None and refpoly.contains(point):              
#                print('point contained in ref',point.coords[0]) 
#                continue
#            if poly_p1.contains(point):              
#                print('point contained in next',point.coords[0]) 
#                continue
#            print('point added',point.coords[0]) 
#            mergedPoly.append(point.coords[0])
#    print("last poly")
#    for point in polys[-1].exterior.coords:
#        point = Point(point)
#        if polys[-2].contains(point):                 
#            print('point contained in previous',point.coords[0]) 
#            continue
#        if refpoly is not None and refpoly.contains(point): 
#            print('point contained in ref',point.coords[0]) 
#            continue
#        print('point added',point.coords[0]) 
#        mergedPoly.append(point.coords[0])
#    u = Polygon(mergedPoly)
     
    return u




class Router():
    def __init__(self, polar, grib, lon0, lat0, lonF, latF, timeStep = 1., limitAzimut = 70., angleStep = 10.):
        self.polar = polar
        self.grib = grib
        self.lon0 = lon0
        self.lat0 = lat0
        self.lonF = lonF
        self.latF = latF
        self.timeStep = timeStep
        self.isochrones = [[(lon0,lat0),]]
        self.isochronesPolygons = list()
        self.isochronesAllPolygons = list()

        self.g = pyproj.Geod(ellps='WGS84')
        
        self.lastGlobalPoly = None
        self.GlobalPolys = list()
        self.azimuteObjectiv, backazimut, self.smallestDistance = self.g.inv(self.lon0, self.lat0, self.lonF, self.latF)
        self.azimuteObjectiv = self.azimuteObjectiv%360
        self.limitAzimut = limitAzimut
        self.limitByDistance = True 
        self.angleStep = angleStep
        self.previous = list()
        self.arrivedWithAzimut = [self.azimuteObjectiv,] #register with wich azimut we arrived to each point of last iso
        self.DistanceParcourueLastIso = None
    
#    def computeIsoForOnePoint(self, lon,lat):
#        windAz,windSpeed = self.grib.interpol(lon,lat)
#        azStart, azEnd = 0., 360
#        points = list()
#        for az in arange(azStart, azEnd, self.angleStep):
#            boatSpeed = self.polar.interpol(windAz-az, windSpeed)
#            dist = boatSpeed*0.514*self.timeStep*3600.
#            #print((windAz-az)%360., boatSpeed, timeStep, dist)
#            lon_f, lat_f, backaz = self.g.fwd(lon,lat,az,dist)
#            points.append((lon_f,lat_f))
#        return points
        
    def computeNextIso(self, log = False):
        time_ini = time.clock()
        polygons = list()
        
        lonlats = array(self.isochrones[-1])
#        windAzs, windSpeeds = self.grib.interpol(lonlats[:,0], lonlats[:,1])
        windAzSpeeds = array([self.grib.interpol(p[0], p[1])  for p in self.isochrones[-1]])
        windAzs, windSpeeds = windAzSpeeds[:,0],  windAzSpeeds[:,1]
        
        for lon, lat, windAz, windSpeed, awa in zip(lonlats[:,0], lonlats[:,1], windAzs, windSpeeds, self.arrivedWithAzimut):
#            points = list()
            azimuteObjectiv, backazimut, smallestDistance = self.g.inv(lon, lat, self.lonF, self.latF)         
            az = arange(self.azimuteObjectiv-120., self.azimuteObjectiv+120., self.angleStep)
#            az = arange(azimuteObjectiv-self.limitAzimut, azimuteObjectiv+self.limitAzimut, self.angleStep)
#            az = arange(awa-self.limitAzimut, awa+self.limitAzimut, self.angleStep)
#            az = arange(awa-100, awa+100, self.angleStep)
#            az = arange(-180., 180., self.angleStep)

            boatSpeeds = array([self.polar.interpol(wa, windSpeed) for wa in windAz-az])
#            print(windSpeed)
#            for a,b in zip(windAz-az,boatSpeeds): print(a,b)
#            stop

            dists = boatSpeeds*0.514*self.timeStep*3600.
            resall = array(self.g.fwd([lon,]*len(az), [lat,]*len(az), az, dists))
            points = resall[0:2,:]
            polygons.append(Polygon(points.transpose()))
#        self.isochronesAllPolygons.append(polygons)
#        print(polygons)
#        print(polygons[0].exterior.coords[:])
        polygon = mergePoly(polygons)
        
        #deduce isoChrone
        if self.lastGlobalPoly is not None:
            iso = [p for p in polygon.exterior.coords if not self.lastGlobalPoly.contains(Point(p[0],p[1]))]
        else:
            iso = polygon.exterior.coords

        self.isochronesPolygons.append(polygon)
#        self.isochronesPolygons.append(Polygon([(self.lon0, self.lat0),]+iso[:]))

            
        if log :
            print('nb of points on calculated iso before filtering',len(iso))

        #register previous
        listToDelet = list()
        previous = list()
        for ip, point in enumerate(iso):
            found = False
            for i,poly in enumerate(polygons):
                if point in poly.exterior.coords : 
#                    self.previous[point] = self.isochrones[-1][i]
                    previous.append(i)
                    found = True
                    break
            # point not in any poly. That mean it is an added intersection. I don't want that
            if not found : listToDelet.append(ip)
        listToDelet.reverse()
        for ip in listToDelet:
#            print("pop",ip)
            iso.pop(ip)

        if log :
            print('nb of points on calculated iso after filtering intersection points',len(iso))

            
            
            
        iso = array(iso)
        previous = array(previous)
        azFromStart = array([self.g.inv(self.lon0, self.lat0, p[0], p[1])[0] for p in iso])
        deltaAzFromObjectiv = (azFromStart-self.azimuteObjectiv+180.+360.)%360.-180.
        isoSort = azFromStart.argsort()
        iso = iso[isoSort]
        previous = previous[isoSort]
        deltaAzFromObjectiv = deltaAzFromObjectiv[isoSort]
        azFromStart = azFromStart[isoSort]
#        print(iso)

        #limit nb of points by azimutLimit
        if self.limitAzimut is not None:
#            filteredIso = [p for p in iso if abs((self.g.inv(self.lon0, self.lat0, p[0], p[1])[0]-self.azimuteObjectiv+180+360)%360 \
#                                          - 180) < self.limitAzimut]
            IsoFilter = abs(deltaAzFromObjectiv)<(self.limitAzimut)
            iso = iso[IsoFilter]
            previous = previous[IsoFilter]
#        print(len(iso))
#        print(iso)
        if log :
            print('nb of points on calculated iso after filtering by azimutlimit',len(iso))

        #recalcul de quelques parametre perdu avec le coup du merge de polynome
        data = array([ self.g.inv(self.isochrones[-1][previous[i]][0], self.isochrones[-1][previous[i]][1], point[0], point[1]) for i,point in enumerate(iso)])
        self.arrivedWithAzimut = data[:,0]
        self.DistanceParcourueLastIso = data[:,2]

            
        #limit nb of point by distance
        if self.limitByDistance and (self.DistanceParcourueLastIso is not None):
#            #compute distance between points
            distanceBetweenIsoPoints = array([self.g.inv(p1[0], p1[1], p2[0], p2[1])[2] for p1,p2 in zip(iso,iso[1:])]+
                                             [self.DistanceParcourueLastIso[-1],]) #ajout d'une valeur par défault pour dernier point pour ne pas le supprimer
            IsoFilter = degrees(arctan(distanceBetweenIsoPoints/self.DistanceParcourueLastIso)) > (self.angleStep*0.5)
#            print(distanceBetweenIsoPoints)
#            print(self.DistanceParcourueLastIso)
#            print(degrees(arctan(distanceBetweenIsoPoints/self.DistanceParcourueLastIso)))
#            print(degrees(arctan(distanceBetweenIsoPoints/self.DistanceParcourueLastIso)) > (self.angleStep/2.))
#            IsoFilter = array([True for p in iso])
            iso = iso[IsoFilter]
            previous = previous[IsoFilter]
            self.arrivedWithAzimut = self.arrivedWithAzimut[IsoFilter]
            self.DistanceParcourueLastIso = self.DistanceParcourueLastIso[IsoFilter]
        if log :
            print('nb of points on calculated iso after filtering by distance',len(iso))
            
            

        iso = [tuple(p) for p in iso]

#        #register previous
#        listToDelet = list()
#        self.arrivedWithAzimut = list()
#        self.DistanceParcourueLastIso = list()
#        for ip, point in enumerate(iso):
#            found = False
#            for i,poly in enumerate(polygons):
#                if point in poly.exterior.coords : 
#                    self.previous[point] = self.isochrones[-1][i]
#                    awa,ba, d = self.g.inv(self.isochrones[-1][i][0], self.isochrones[-1][i][1], point[0], point[1])
#                    self.arrivedWithAzimut.append(awa)
#                    self.DistanceParcourueLastIso.append(d)
#                    found = True
#                    break
#            # point not in any poly. That mean it is an added intersection. I don't want that
#            if not found : listToDelet.append(ip)
#        listToDelet.reverse()
#        for ip in listToDelet:
##            print("pop",ip)
#            iso.pop(ip)
        
#        self.DistanceParcourueLastIso = array(self.DistanceParcourueLastIso)    
        self.isochrones.append(iso)
#        iso.extend([(self.lon0, self.lat0),])
        self.lastGlobalPoly = Polygon(iso+[(self.lon0, self.lat0),])
        self.GlobalPolys.append(self.lastGlobalPoly)
        self.previous.append(previous)

        if log :
            print('nb of points on calculated iso',len(iso))
            print('time computing iso : ', time.clock()-time_ini)
        
    def plotSteps(self):
        fig, axis = plt.subplots(3,3,figsize=(10, 10), sharex=True, sharey=True)
        fig.subplots_adjust(hspace=0, wspace = 0)
        axis = [ax for axline in axis for ax in axline ]

        routeDirect = array([(self.lon0,self.lat0),]+self.g.npts(self.lon0, self.lat0, self.lonF, self.latF, 20))
        routeDirectLon = routeDirect[:,0]
        routeDirectLat = routeDirect[:,1]

        # plot limitLeft
        limitLeftAz = self.azimuteObjectiv-self.limitAzimut
        limitLeft_lonf, limitLeft_latf, backaz = self.g.fwd(self.lon0, self.lat0, limitLeftAz, self.smallestDistance)
        routeDirect = array([(self.lon0,self.lat0),]+self.g.npts(self.lon0, self.lat0, limitLeft_lonf, limitLeft_latf, 20))
        routeLimitLeftLon = routeDirect[:,0]
        routeLimitLeftLat = routeDirect[:,1]
        # plot limitright
        limitRightAz = self.azimuteObjectiv+self.limitAzimut
        limitRight_lonf, limitRight_latf, backaz = self.g.fwd(self.lon0, self.lat0, limitRightAz, self.smallestDistance)
        routeDirect = array([(self.lon0,self.lat0),]+self.g.npts(self.lon0, self.lat0, limitRight_lonf, limitRight_latf, 20))
        routeLimitRightLon = routeDirect[:,0]
        routeLimitRightLat = routeDirect[:,1]
        
        for i,ax in enumerate(axis):
            if i >= len(self.isochrones) : break
            print('plot',i)
            ax.set_xlim([-45.2,-44])
            ax.set_ylim([-45.2,-44])

            ax.plot(routeDirectLon,routeDirectLat)
            ax.plot(routeLimitLeftLon,routeLimitLeftLat)
            ax.plot(routeLimitRightLon,routeLimitRightLat)

            if i > 1: 
                polyglobpatch = PolygonPatch(self.GlobalPolys[i-2], fc='b', ec='b', alpha=0.7, zorder=2)
                ax.add_patch(polyglobpatch)

            for ii in range(i):
                a = array(self.isochrones[ii])
                ax.plot(a[:,0],a[:,1],'-o')

            
            if not i == 0: 

                #for poly_i in self.isochronesAllPolygons[i-1]:
                #    a = array(poly_i.exterior.coords)
                #    ax.plot(a[:,0],a[:,1],'-o')
                #a = array(self.isochronesAllPolygons[i-1][0].exterior.coords)
                #ax.plot(a[:,0],a[:,1],'-o')
                #a = array(self.isochronesAllPolygons[i-1][-1].exterior.coords)
                #ax.plot(a[:,0],a[:,1],'-o')                
                polyglobpatch = PolygonPatch(self.isochronesPolygons[i-1], fc='r', ec='r', alpha=0.3, zorder=2)
                ax.add_patch(polyglobpatch)
                #polyglobpatch = PolygonPatch(self.isochronesPolygons[i-1].simplify(0.001, preserve_topology=True), fc='r', ec='r', alpha=0.3, zorder=2)
                #ax.add_patch(polyglobpatch)
            a = array(self.isochrones[i])
            ax.plot(a[:,0],a[:,1],'-o')

#            if not i == 0: 
#                a = array(self.isochronesPolygons[i-1].simplify(0.001, preserve_topology=True).exterior.coords)
#                ax.plot(a[:,0],a[:,1],'-o')
            
            for ip, point in enumerate(self.isochrones[i]):
                prev = self.isochrones[i-1][self.previous[i-1][ip]]
                if prev is not None:
                    ax.plot([point[0],prev[0]],[point[1],prev[1]],'g')
        fig.show()
    
grib = GRIB()      
#grib.plot()        
polaire = POLAR("C:/Users/clopeau/Documents/etienne/python/routage_voile/polaires/polaire_exemple.csv")
#polaire.plot()

router = Router(polaire, grib, lon0, lat0, latF, lonF, timeStep, limitAzimut=60, angleStep = 5)
for i in range(8):
    print(i)
    router.computeNextIso(log = True)
#router2 = Router(lon0, lat0, latF, lonF, timeStep, limitAzimut=None)
#for i in range(3):
#    print(i)
#    router2.computeNextIso()


#fig, ax = plt.subplots(3,3,figsize=(7, 7))
#for i,iso in enumerate(router.isochrones):
#    if not i == 0: 
#        p = router.isochronesPolygons[i-1].exterior.coords
#        a = array(p)
#        ax.plot(a[:,0],a[:,1],'-o')
#    a = array(iso)
#    ax.plot(a[:,0],a[:,1],'-o')

router.plotSteps()

