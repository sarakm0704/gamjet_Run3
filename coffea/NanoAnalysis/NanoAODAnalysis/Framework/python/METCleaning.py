def METCleaning(events,year):
    if '2016' in year:
        return (events.Flag.goodVertices &
                events.Flag.globalSuperTightHalo2016Filter &
                events.Flag.HBHENoiseFilter &
                events.Flag.HBHENoiseIsoFilter &
                events.Flag.BadPFMuonFilter &
                events.Flag.eeBadScFilter
        )

    return (events.Flag.goodVertices &
            events.Flag.globalSuperTightHalo2016Filter &
            events.Flag.HBHENoiseFilter &
            events.Flag.HBHENoiseIsoFilter &
            events.Flag.BadPFMuonFilter &
            events.Flag.ecalBadCalibFilter
    )
