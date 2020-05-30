import { Component, OnInit } from '@angular/core';
import { HttpService } from './common/http.service';
import { CommonService } from './common/common.service';
import { CubeService } from './common/cube.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styles: [
    "button.btn-action { width: 65px; }"
  ],
})
export class AppComponent implements OnInit {

  scrambleDepth = 10;

  constructor(public commonService: CommonService, public cubeService: CubeService, public httpService: HttpService) { }

  ngOnInit() {
    this.getInitData();
  }

  private async getInitData() {
    await this.commonService.getInfo();
    this.commonService.status.hasData = true;
  }
}
