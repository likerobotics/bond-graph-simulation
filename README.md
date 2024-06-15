# Bond Graph simulation (python based)

This package allows:
+ to create a bond-graph model of robotics and mechatronics systems (Electrical, Mechanical, Thermal domains).
+ to obtain state-space form of the system with energy variables as a state variables
+ make a simulation of the system

## Restrictions:
Curent version does work only with the 1-D systems.
Examples mostly focuses on Robotics domain.

Directory is organized as folows:
- run/BondGraph.py - the core of bondgraph models
- run/utils.py - the useful functionality to help with model creation
- examples/* - contains some scripts with the simple example to help beginers
- dev/* - contains some scripts under development

Model *(BondGraph) architecture:
- Any Port must belong to some Element (Element has a list of associated ports)
- Several Ports can belong to one Element
- Any Bond must connected only 2 Ports (Bond holds the connected ports)
- 

### Dependencies shown in the requiremens.txt


### Software architecture
- class BGport
    + id
    + parent
    + type
    + name
    + position
    + value
    + direction
    + causality
    + effort
    + flow
- class BGelement
    + id
    + __type
    + __name
    + ports
    + effort
    + flow
    + stateVariable
    + inputVariable
    + parameter
    + stateEquation
    + outputEquation
- class BGbond
    + id
    + __fromPort
    + __toPort
    + effort
    + flow
    + __causalityStroke
    + __directionArrow
    + __type
- class BondGraph
    + name
    + bondsList
    + elementsList
    + portsList
    + state_vector

    + methods:
        + addElement
        + addBond
        + getBondList
        + getPortsList
        + render
          + set the appropriate positions for drawing.


### BG equastions asigning sequence
1. Assign equations for 1 and 0 junction's ports (for the SE, SF, I, C and R nodes we can predefine the equastions)
2. Assign equations for the each bonds (except 1 to 0 and 0 to 1 connections)
3. Asign equations for 1 to 0 and 0 to 1 junctions

## Errors and Warnings
001: Can not apply rules (input-output and causality)
002: ReExpression error: can not find the variable in the given expretion
003: Probably not defined causality during construction of the equation systems
004: Error during making the equations same form, probably some sign lost
005: Some causality lost in drawings
006: 
007: Error, The bond has no effort or flow
008: 

### TODO list
Add example of inverted pendulum on card
Add GY elements
GUI to draw the bondgraph (without coding)
API for easy interaction with model


#### How to contribute
If you are interested to become a contributor, please send me an email. 

Useful tools to work with state-space form:

https://github.com/python-control/python-control

https://python-control.readthedocs.io/en/0.9.3.post2/

