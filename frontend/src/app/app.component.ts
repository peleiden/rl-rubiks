import { Component, OnInit } from '@angular/core';
import { HttpService } from './common/http.service';
import { CommonService } from './common/common.service';
import { CubeService } from './common/cube.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
})
export class AppComponent implements OnInit {

  constructor(public commonService: CommonService, public cubeService: CubeService, public httpService: HttpService) { }

  ngOnInit() {
    this.commonService.setSelectedHost(this.httpService.selectedHost);
  }
}
