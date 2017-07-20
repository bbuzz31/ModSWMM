 !
 !FUNCTIONS FOR PAUSING MODFLOW
 !BB 09.27.16
 !I think I need to convert this f77
 !  -------------------------------------------------------------------

      INTEGER FUNCTION LENS(STR)
      !
      CHARACTER*(*) STR
      !
      lens=len(str)
      IF (lens.gt.1) THEN
          DO WHILE (lens.gt.1.and.STR(lens:lens).eq.' ');
            LENS=lens-1
          END DO
      ENDIF

      IF (lens.eq.1.and.str(lens:lens).eq.' ') lens = 0
      END FUNCTION LENS

!---------------------------------------------------------------------

      LOGICAL FUNCTION swmm_is_done(tstep, delay)

      INTEGER delay, getcwd, status, file_exists, tstep
      CHARACTER*120 cwd
      CHARACTER*50 filename, make_file_name

      ! make the swmm filename
      ! needs previous time step (-1); looks for swmm00001 to start mf step 2
      filename = make_file_name('swmm_done', tstep-1, '.txt')
      status = getcwd( cwd )

      ! WRITE(*, 99) tstep, TRIM(cwd), TRIM(filename)
! 99   FORMAT(/,5X,' MF step ',I5, ' waiting for: ', A, '/', A/)

      swmm_is_done = .FALSE.
      !print *, 'Looking for: ', filename, ' in ', cwd
      if( status .ne. 0 ) stop 'getcwd: error'
      DO WHILE (.NOT.swmm_is_done)
        file_exists = ACCESS(filename, " ")
        IF (file_exists .EQ. 0) THEN
          ! eventually have file deleted and increment n in main prog
          swmm_is_done= .TRUE.

        ELSE
          CALL SLEEP(delay)
        ENDIF

      END DO
     END FUNCTION swmm_is_done

!---------------------------------------------------------------------

      LOGICAL FUNCTION modflow_is_done(root, kper)

      INTEGER kper
      CHARACTER*80  filename, make_file_name
      CHARACTER*(*) root
      filename = make_file_name(root, kper, '.txt')
      OPEN (unit=110, file=filename, status='NEW', iostat=ios)
      CLOSE (unit=110, status='keep')

      ! print*, 'kper', kper, 'is done'
      RETURN

      END FUNCTION modflow_is_done

 ! --------------------------------------------------------------------


      CHARACTER*(*) FUNCTION make_file_name(string, n, ext)

      ! filename string, filenum extension n, ext
      CHARACTER*(*) string, ext
      INTEGER n

      ! careful with this strlength
      ! CHARACTER(LEN=200), allocatable :: fileplace

      CHARACTER*5 CINT

      !fileplace = 'C:\Users\bbuzzang\Dropbox\Thesis_scripts\fortran\&
    !& function_testing\function_testing1\x64\Debug\'

      make_file_name = string(1:LENS(string))//CINT(n,5)//ext
           DO i=LENS(string)+1,LENS(string)+5
              IF (make_file_name(i:i).eq.' ') make_file_name(i:i)='0'
           END DO


 END FUNCTION make_file_name

! ---------------------------------------------------------------------

      CHARACTER*(*) FUNCTION CINT (I,M)
      ! integer to character in form of 0000I
      INTEGER I,M
      CHARACTER FORM*10
      ! Write m to form using format 100
      WRITE (FORM,100) M
      WRITE (CINT,FORM) I
      ! I2 = integer, 2 positions
      100 FORMAT('(I',I2,')')

 END FUNCTION CINT

! ---------------------------------------------------------------------
! OPEN/CLOSE  C:/Users/bbuzzang/Google_Drive/WNC/Flopy/second_test/externals\finf_2.ref               1  (10E15.6) -1 finf_2
