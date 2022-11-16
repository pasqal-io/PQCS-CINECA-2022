using JSON

struct Coordinate
    x::Float64
    y::Float64
    name::String
end

struct PulseInfo
    omega_start::Float64
    omega_stop::Float64
    delta_start::Float64
    delta_stop::Float64
    duration::Float64
end

struct LaserParam
    omega::Float64
    delta::Float64
end

struct DataJSON
    coordinates::Vector{Coordinate}
    pulse::Vector{PulseInfo} #array of pulse params
end

"""
parse_registers(registers::Vector{Any}) -> Vector{Coordinate}
registers come with their X,Y coordinates and name "qbit_name"
parse_registers parses coordinate to struct Coordinate 
"""
function parse_registers(registers::Vector)
    coordinates = Vector{Coordinate}()
    for qbit in registers
        push!(coordinates, Coordinate(qbit["x"], qbit["y"], qbit["name"]))
    end
    return coordinates
end

"""
parse_operation_info(operation::Dict{String, Any}) -> {start::Float, stop::Float, duration::Float}
Helper function in parse_operation()
Parses JSON pulse data to pulse {begin, start, pulse duration}
"""
function parse_operation_info(operation::Dict{String,Any})
    #ADD version with 
    if operation["kind"] == "ramp"
        return operation["start"], operation["stop"], operation["duration"]
    elseif operation["kind"] == "constant"
        return operation["value"], operation["value"], operation["duration"]
    else
        throw(ArgumentError("The pulse should be of kind 'ramp' or 'constant'."))
    end
    return nothing
end

"""
parse_operation(operations::Vector{Any}) -> Vector{PulseInfo}
Parses JSON pulse data to array of pulses {begin, start, pulse duration}
"""
function parse_operation(operations::Vector)
    pulseinfo = Vector{PulseInfo}()
    for operation in operations
        if operation["op"] == "pulse"
            omegastart, omegastop, omegaduration =
                parse_operation_info(operation["amplitude"])
            deltastart, deltastop, deltaduration =
                parse_operation_info(operation["detuning"])
            if omegaduration != deltaduration
                throw(
                    ArgumentError(
                        "Amplitude and Detuning part of the pulse must have identical duration.",
                    ),
                )
            end
            timetotal = omegaduration
            pulsetmp = PulseInfo(omegastart, omegastop, deltastart, deltastop, timetotal)
            push!(pulseinfo, pulsetmp)
        end
    end
    if isempty(pulseinfo)
        throw(ArgumentError("No supported operations were provided."))
    end
    return pulseinfo
end

"""
parse_json(jsonpath::String)
Parses JSON into two arrays of coordinates and pulses
"""
function parse_json(jsonpath::String)
    rootJSON = JSON.parsefile(jsonpath)
    coordinates = parse_registers(rootJSON["register"])
    pulseinfo = parse_operation(rootJSON["operations"])
    return DataJSON(coordinates, pulseinfo)
end

"""
write_json(result, path) -> nothing
writes results of simulation into JSON file.
"""
function write_json(result, path)
    open(path, "w") do f
        return JSON.print(f, result, 4)
    end
    return nothing
end

"""
discretize(pulseinfo::PulseInfo, dt::AbstractFloat) -> Vector{LaserParam}
Helper function of parse()
Converts pulse data {begin, end, duration} into discretized array of {omega, delta}
"""
function discretize(pulseinfo::PulseInfo, dt::AbstractFloat)
    omega_start = pulseinfo.omega_start
    omega_stop = pulseinfo.omega_stop
    delta_start = pulseinfo.delta_start
    delta_stop = pulseinfo.delta_stop
    #=
        I found that the evolution operator has inconsistent units of time.
        [time] = nanoseconds,
        [Hamiltonian] = [Omega] = 1/microseconds.
        In the real applications we use pulses >1000 nanoseconds.
        1000 nanoseconds = 1 microsecond and therefore pulse time duration should be multiplied by 1e-3.
        To make exp[ -i * time * Hamiltonian] unitless, one has to convert nanoseconds to microseconds.
    =#

    timetotal = pulseinfo.duration

    container = Vector{LaserParam}()
    omega_rate = (omega_stop - omega_start) / timetotal
    delta_rate = (delta_stop - delta_start) / timetotal

    omega_step = omega_rate * dt
    delta_step = delta_rate * dt
    nsteps = Int(round(timetotal / dt))
    ##DISCRETIZATION IS HERE ! in the step of for loop
    for n in 1:nsteps
        omega = omega_start + omega_step * n
        delta = delta_start + delta_step * n
        push!(container, LaserParam(omega, delta))
    end
    return container
end

"""
discretize(pulses::Vector{PulseInfo}, discrstep::AbstractFloat) -> Vector{LaserParam}
Converts pulse data ARRAY{begin, end, duration} into discretized array of {omega, delta}
"""
function discretize(pulses::Vector{PulseInfo}, discrstep::AbstractFloat)
    pulse = Vector{LaserParam}()
    for pulseinfo in pulses
        pulsetmp = discretize(pulseinfo, discrstep)
        append!(pulse, pulsetmp)
    end
    return pulse
end
