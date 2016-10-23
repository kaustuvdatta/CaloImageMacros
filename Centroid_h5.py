# coding: utf-8
# Kaustuv Datta and Jayesh Mahaptra, July 2016 (+Niki Howe, May 2016)
# pass in a string of events as a parameter, and it will parse and centre the events

import numpy as np
import sys
import ast
import h5py

# Given a list of distances and corresponding weights, calculate the weighted average
def findMidpoint(distance, energy):
    return np.average(distance, weights = energy)
    
# Given an array of interactions (ix,iy,iz,E,X,Y,Z), returns the weighted average (aveY, aveZ)
def findEventMidpoint(event):
    Yave = findMidpoint(event[:,5], event[:,3]) 
    Zave = findMidpoint(event[:,6], event[:,3]) 
    return (Yave, Zave)
    
# Check if between bounds

#Checking range for ECAL (including min and max as the total number of cells is odd)
def withinEcal(value, mymin, mymax):
    return (value >= mymin and value <= mymax)

#Checking range for HCAL (including min but excluding max as the number of cells is even)
def withinHcal(value, mymin, mymax):
    return (value >= mymin and value < mymax)

# Given an event, get the 20x20x25 array of energies around its barycentre
def getECALArray(event):
    
    # Get the event midpoint (returns absolute Y,Z weighted average at index 0,1 respectively)
    midpoint = findEventMidpoint(event)
    
    #Map absolute Y  and Z weighted average to ix and iy respectively
    #Rounding the mapping to get barycenter ix,iy values as integer (in order to select a particular cell as barycenter)
    barycenter_ix = round(midpoint[0]/5)
    barycenter_iy = round(midpoint[1]/5)
    
    # Get the limit points for our grid
    xmin = barycenter_ix - 12
    xmax = barycenter_ix + 12
    ymin = barycenter_iy - 12
    ymax = barycenter_iy + 12
    
    # Create the empty array to put the energies in
    final_array = np.zeros((25, 25, 25))
    
    # Fill the array with energy values, if they exist
    #for element in event:
    #    if within(element[0], xmin, xmax) and within(element[1], ymin, ymax) and within(element[2], zmin, zmax):
    #        final_array[element[0], element[1], element[2]] = element[3]
  
    # Fill the array with energy values, if they exist
    for ix, iy, iz, E, x, y, z in event:
        if withinEcal(ix, xmin, xmax) and withinEcal(iy, ymin, ymax):
            final_array[ix-xmin,iy-ymin,iz] = E
    return final_array,midpoint[0],midpoint[1]

# Given an event and Absolute Y and Z coordinates of the ECAL centroid
# get the 4x4x60 array of energies around the same coordinates of HCAL
def getHCALArray(event,midpointY,midpointZ):
    
    # Use the Y and Z of ECAL centroid
   
    #Map absolute Y and Z weighted average to ix and iy respectively
    #Rounding the mapping to get barycenter ix,iy values as integer (in order to select a particular cell as barycenter)
    barycenter_ix = round(midpointY/30)
    barycenter_iy = round(midpointZ/30)
    
    # Get the limit points for our grid
    xmin = barycenter_ix - 2
    xmax = barycenter_ix + 2
    ymin = barycenter_iy - 2
    ymax = barycenter_iy + 2
    
    # Create the empty array to put the energies in
    final_array = np.zeros((4, 4, 60))
    
    # Fill the array with energy values, if they exist
    for ix, iy, iz, E, x, y, z in event:
        if withinHcal(ix, xmin, xmax) and withinHcal(iy, ymin, ymax):
            final_array[ix-xmin,iy-ymin,iz] = E
    return final_array

def convertFile(filename):
    #The main testing cell
    with open(filename) as myfile:
            my_events_string = myfile.read().replace('\n', '')
    my_events_string = my_events_string.replace(' ', '')
    my_events_string = my_events_string.replace('}{','} {')
    my_events_string = my_events_string.split()
    ECAL_array_list = []
    HCAL_array_list = []
    target_array_list = []

#loop through all the events
    for string in my_events_string:
        my_event = ast.literal_eval(string)
        

#Make a list containing all the cell readouts of ECAL for the event and store it in a single ECAL array
        ECAL_list = []
        for cell_readout in my_event['ECAL']:
            ECAL_list.append(np.array(cell_readout))

#Store all the cell readings to make a ECAL readout array for this particular event
        ECAL_array = np.array(ECAL_list)

#Get the barycenter details (The ECAL array around barycenter, Absolute Y and Z coordinates of ECAL centroid at index 0,1,2 respectively)
        ECAL_barycenter_details = getECALArray(ECAL_array)

#Append the ECAL array of 25x25x25 cells around the barycenter to the ECAL array list
        ECAL_array_list.append(ECAL_barycenter_details[0])

#Make a list containing all the cell readouts of HCAL for the event and store it in a single ECAL array
        HCAL_list = []
        for cell_reading in my_event['HCAL']:
            HCAL_list.append(np.array(cell_reading))

#Store all the cell readings to make a HCAL readout array for this particular event
        HCAL_full_array = np.array(HCAL_list)

#Pass the absolute Y and Z cooridnates as input for determining HCAL array around barrycenter and append it to the HCAl array list
        HCAL_array_list.append(getHCALArray(HCAL_full_array,ECAL_barycenter_details[1],ECAL_barycenter_details[2]))

#collecting particle ID, energy of hit, and 3-vector of momentum
        pdgID = my_event['pdgID']
        if pdgID == 211 or pdgID == 111:
            pdgID = 0
        if pdgID == 22:
            pdgID = 1
        energy = my_event['E']
        energy = energy/1000.
        (px, py, pz) = (my_event['px'], my_event['py'], my_event['pz'])
        (px, py, pz) = (px/1000., py/1000., pz/1000.)
        target = np.zeros((1, 5))
        target[:,0], target[:,1], target[:,2], target[:,3], target[:,4] = (pdgID, energy, px, py, pz)
        target_array_list.append(target)

#Save the final ECAL and HCAL array list to an h5 file
    f = h5py.File(filename.replace('.txt','.h5'), "w")
    f.create_dataset('ECAL',data=np.array(ECAL_array_list),compression='gzip')
    f.create_dataset('HCAL',data=np.array(HCAL_array_list),compression='gzip')
    f.create_dataset('target',data=np.array(target_array_list),compression='gzip')
    f.close()

    
if __name__ == "__main__":
    import sys
    convertFile(sys.argv[1])